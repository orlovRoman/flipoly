import dataclasses
import json
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory, MarketSnapshot
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision, MarketSignal, decide_favorite, decide_crypto_trend, decide_ml_trend, decide_outsider
from polyflip.trading.ml_inference import build_inference_dataframe, run_model_inference
from polyflip.crypto.predictor import MIN_CANDLES_REQUIRED
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.trading.utils import compute_dead_zone
from polyflip.trading.funnel_logger import log_funnel

logger = structlog.get_logger(__name__)

def _get_float_setting(raw_settings: dict, key: str) -> Optional[float]:
    """
    Читает float из raw_settings.
    Возвращает None, если значение отсутствует или пустое.
    """
    val = raw_settings.get(key)
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None

@dataclass
class DecisionResult:
    decision_obj: Optional[TradeDecision]
    p_flip: float
    model_ver: Optional[int]
    edge: Optional[float]
    skip_reason: Optional[str]
    lgbm_metadata: Optional[str] = None
    used_model_key: Optional[str] = None
    confirm_model_key: Optional[str] = None
    confirm_model_version: Optional[int] = None
    applied_lower: Optional[float] = None
    applied_upper: Optional[float] = None

async def decide_favorite_mode(
    market: LiveMarket,
    cfg: TradingConfig,
    asset_min_edge: float,
    asset_max_price: float,
    start_time: datetime,
    time_left_sec: float,
) -> DecisionResult:
    if market.current_yes_price == 0.5:
        logger.info("favorite_mode_skip_no_favorite", market_id=market.market_id)
        return DecisionResult(None, 0.0, None, None, "Pure Favorite: no clear favorite (price == 0.5)")

    signal = MarketSignal(
        asset=market.asset,
        mid_price=market.current_yes_price,
        spread=market.current_spread or 0.01,
        volume_5min=market.volume_5min or 0.0,
        price_velocity=market.price_velocity or 0.0,
        hour_of_day=start_time.hour,
        time_left_min=time_left_sec / 60.0,
        yes_bid=getattr(market, "current_yes_bid", None),
        yes_ask=getattr(market, "current_yes_ask", None),
        no_bid=getattr(market, "current_no_bid", None),
        no_ask=getattr(market, "current_no_ask", None),
    )
    
    local_fav_config = {
        "FAVORITE_THRESHOLD": str(cfg.favorite_threshold),
        "FAVORITE_MIN_EDGE": str(cfg.favorite_min_edge),
        "DEAD_ZONE_WIDTH": str(cfg.dead_zone),
        "FAVORITE_MIN_PRICE": str(cfg.favorite_min_price),
        "FAVORITE_MAX_PRICE": str(cfg.favorite_max_price),
        "TRADE_BET_SIZE_USDC": str(cfg.bet_size),
        "MAX_BET_SIZE_USDC": str(cfg.max_bet_size_usdc),
        "MIN_EDGE": str(asset_min_edge),
        "TRADE_MAX_PRICE": str(asset_max_price),
        "LIQUIDITY_FRACTION": str(cfg.liquidity_fraction),
        "BET_SIZING_MODE": str(cfg.bet_sizing_mode),
    }
    
    decision_obj = decide_favorite(signal, local_fav_config)
    if not decision_obj.decision_details:
        decision_obj = dataclasses.replace(decision_obj, decision_details={"market_role": "FAVORITE"})
    if not cfg.trade_on_favorite:
        decision_obj = dataclasses.replace(decision_obj, action="SKIP", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)")
    
    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=0.0,
        model_ver=None,
        edge=decision_obj.edge,
        skip_reason=decision_obj.reason if decision_obj.action == "SKIP" else None
    )

async def infer_flip_for_market(
    db_session: AsyncSession,
    market: LiveMarket,
    model: Any,
    active_features: list[str],
    fresh_price: float,
    fresh_spread: float,
    start_time: datetime,
    time_left_sec: float,
    max_time_left: float,
) -> float:
    from datetime import timedelta
    cutoff_time = start_time - timedelta(seconds=max_time_left)

    snapshots_stmt = select(MarketSnapshot).where(
        MarketSnapshot.market_id == market.market_id,
        MarketSnapshot.recorded_at >= cutoff_time
    ).order_by(MarketSnapshot.recorded_at.asc())
    snapshots_res = await db_session.execute(snapshots_stmt)
    history_snaps = snapshots_res.scalars().all()

    filtered_prices = [
        float(s.mid_price) for s in history_snaps
        if s.recorded_at >= cutoff_time
    ] + [fresh_price]
    global_max = max(filtered_prices) if filtered_prices else fresh_price

    df = build_inference_dataframe(
        market=market,
        history_snaps=list(history_snaps), 
        fresh_yes_price=fresh_price,
        fresh_spread=fresh_spread,
        global_max=global_max,
        start_time=start_time,
        time_left_sec=time_left_sec,
    )
    
    return float(run_model_inference(df, model, active_features))

async def _get_funding_rate(db_session: AsyncSession, binance_symbol: str) -> float | None:
    from sqlalchemy import select
    from polyflip.db.models import RuntimeSettings
    from datetime import datetime, timezone, timedelta
    fr_key = f"FUNDING_RATE_{binance_symbol}"
    try:
        row = (await db_session.execute(select(RuntimeSettings).where(RuntimeSettings.key == fr_key))).scalar_one_or_none()
        if not row:
            return None
        if row.updated_at and row.updated_at.tzinfo is None:
            row.updated_at = row.updated_at.replace(tzinfo=timezone.utc)
        if row.updated_at and datetime.now(timezone.utc) - row.updated_at > timedelta(hours=12):
            return None
        return float(row.value)
    except Exception:
        return None

async def decide_ml_mode(
    db_session: AsyncSession,
    api_client: Any,
    market: LiveMarket,
    cfg: TradingConfig,
    raw_settings: Optional[dict] = None,
    models_cache: Optional[Any] = None,
    crypto_predictor: Optional[Any] = None,
    start_time: Optional[datetime] = None,
    time_left_sec: float = 600.0,
    existing_skipped: Optional[TradeHistory] = None,
    execution_mode: str = "PAPER",
) -> DecisionResult:
    if raw_settings is None:
        raw_settings = {}
    if start_time is None:
        start_time = datetime.now(timezone.utc)
    if not models_cache or not models_cache.models:
        logger.warning("decide_ml_mode_empty_cache_skip", asset=market.asset)
        return DecisionResult(None, 0.0, None, None, "Models cache is empty")
    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
        error_msg = fresh_yes_prices.get("error", "No fresh YES prices from API") if fresh_yes_prices else "No fresh YES prices from API"
        return DecisionResult(None, 0.0, None, None, error_msg)

    fresh_yes_price = fresh_yes_prices["current_yes_price"]
    fresh_spread = fresh_yes_prices.get("current_spread", market.current_spread)
    
    from polyflip.constants import get_price_phase
    phase = get_price_phase(fresh_yes_price)
    phase_asset = f"{market.asset.upper()}_{phase}"

    if phase_asset in models_cache.models:
        model = models_cache.models[phase_asset]
        model_ver = models_cache.versions.get(phase_asset)
        active_features = models_cache.features.get(phase_asset, [])
        used_model = phase_asset
    else:
        model = models_cache.models.get(market.asset.upper())
        model_ver = models_cache.versions.get(market.asset.upper())
        active_features = models_cache.features.get(market.asset.upper(), [])
        used_model = market.asset.upper()
    
    logger.info(
        "ml_model_selected",
        asset=market.asset,
        fresh_price=round(fresh_yes_price, 4),
        phase=phase,
        phase_asset=phase_asset,
        phase_available=(phase_asset in models_cache.models),
        used_model=used_model,
    )
    
    if not model:
        return DecisionResult(None, 0.0, None, None, f"No active model found for {market.asset.upper()}")
        
    try:
        p_flip = await infer_flip_for_market(
            db_session=db_session,
            market=market,
            model=model,
            active_features=active_features,
            fresh_price=fresh_yes_price,
            fresh_spread=fresh_spread,
            start_time=start_time,
            time_left_sec=time_left_sec,
            max_time_left=cfg.max_time_left,
        )
    except Exception as e:
        logger.error("ml_mode_infer_flip_error", asset=market.asset, error=str(e))
        return DecisionResult(None, 0.0, None, None, "flip inference failed")

    # Диагностика: сырое значение p_flip до любых коррекций (ECE, dead zone и т.д.)
    # Если is_suspicious=True (значение близко к 0.5) — проблема в фичах, а не в порогах.
    logger.info(
        "p_flip_raw_pre_correction",
        asset=market.asset,
        market_id=market.market_id,
        p_flip_raw=round(p_flip, 6),
        fresh_price=round(fresh_yes_price, 4),
        used_model=used_model,
        n_active_features=len(active_features),
        is_suspicious=(abs(p_flip - 0.5) < 0.01),
    )

    signal = MarketSignal(
        asset=market.asset,
        mid_price=fresh_yes_price,
        spread=fresh_spread or 0.01,
        volume_5min=market.volume_5min or 0.0,
        price_velocity=market.price_velocity or 0.0,
        hour_of_day=start_time.hour,
        time_left_min=time_left_sec / 60.0,
        yes_bid=getattr(market, "current_yes_bid", None),
        yes_ask=getattr(market, "current_yes_ask", None),
        no_bid=getattr(market, "current_no_bid", None),
        no_ask=getattr(market, "current_no_ask", None),
    )

    _asset_upper = market.asset.upper()
    _priority_chain = [
        f"FLIP_THRESHOLD_{_asset_upper}",              # per-asset из дашборда
        "FLIP_THRESHOLD",                              # глобальный из дашборда
        f"AUTO_FLIP_THRESHOLD_{used_model}_contested",  # авто: фаза contested
        f"AUTO_FLIP_THRESHOLD_{used_model}_leaning",    # авто: фаза leaning
        f"AUTO_FLIP_THRESHOLD_{used_model}_decided",    # авто: фаза decided
        f"AUTO_FLIP_THRESHOLD_{used_model}",            # авто: базовый для модели
        f"AUTO_FLIP_THRESHOLD_{_asset_upper}",          # авто: по символу
    ]

    base_flip_threshold = None
    _resolved_key = None
    for _key in _priority_chain:
        _val = _get_float_setting(raw_settings, _key)
        if _val is not None:
            base_flip_threshold = _val
            _resolved_key = _key
            break

    if base_flip_threshold is None:
        base_flip_threshold = cfg.no_flip_threshold + cfg.dead_zone
        _resolved_key = "system_default"

    if base_flip_threshold > 1.0:
        logger.warning(
            "flip_threshold_normalized_from_percent",
            resolved_key=_resolved_key,
            original=base_flip_threshold,
            normalized=base_flip_threshold / 100.0,
        )
        base_flip_threshold /= 100.0

    logger.info(
        "flip_threshold_resolved",
        asset=_asset_upper,
        resolved_key=_resolved_key,
        value=round(base_flip_threshold, 4),
        used_model=used_model,
    )

    lower, upper = compute_dead_zone(
        flip_threshold=base_flip_threshold,
        dead_zone_width=cfg.dead_zone,
        auto_mode=cfg.auto_dead_zone,
    )

    local_config = {**raw_settings}
    local_config["NO_FLIP_THRESHOLD"] = str(lower)
    local_config["FLIP_THRESHOLD"] = str(upper)
    # В ML_TREND режиме FAVORITE_THRESHOLD фиксирован в 0.50:
    # mid >= 0.50 → YES-фаворит, mid < 0.50 → NO-фаворит.
    # Dead zone [lower, upper] отсекается отдельно через is_in_dead_zone.
    local_config["FAVORITE_THRESHOLD"] = "0.50"
    # В ML_TREND режиме decide_favorite используется только для определения стороны (YES/NO),
    # но не для фильтрации по edge — это делает decide_ml_trend с MIN_EDGE.
    # Поэтому передаём заведомо низкий порог, чтобы favorite не заблокировал стороны
    # из-за своего edge-фильтра. Итоговый edge проверяется в decide_ml_trend.
    ML_MODE_BYPASS_FAV_EDGE = -100.0
    local_config["FAVORITE_MIN_EDGE"] = str(ML_MODE_BYPASS_FAV_EDGE)
    local_config["BYPASS_BET_SIZE_CHECK"] = "true"

    logger.info(
        "local_config_for_decision",
        asset=_asset_upper,
        flip_threshold=round(base_flip_threshold, 4),
        favorite_threshold=0.50,
        no_flip_threshold=round(lower, 4),
        fav_min=local_config.get("FAVORITE_MIN_PRICE"),
        fav_max=local_config.get("FAVORITE_MAX_PRICE"),
        dead_zone=local_config.get("DEAD_ZONE_WIDTH"),
        no_min_edge=local_config.get("NO_MIN_EDGE"),
    )

    ece = getattr(models_cache, "eces", {}).get(used_model, 0.0)
    from polyflip.trading.position_sizing import apply_ece_correction
    calibrated = apply_ece_correction(p_flip, ece)
    logger.info(
        "ml_ece_diagnostics",
        asset=market.asset,
        ece=round(ece, 4),
        p_flip_raw=round(p_flip, 4),
        p_flip_calibrated=round(calibrated, 4),
        calibration_shift=round(abs(p_flip - calibrated), 4),
        warn=(ece > 0.05),
    )

    if cfg.trade_on_favorite:
        decision_obj = decide_ml_trend(signal, p_flip, local_config, ece=ece)
        if decision_obj.action == "SKIP" and not decision_obj.decision_details:
            decision_obj = dataclasses.replace(decision_obj, decision_details={"market_role": "FAVORITE"})
    else:
        decision_obj = TradeDecision(action="SKIP", buy_price=0.0, bet_size_usdc=0.0, strategy_type="ML_TREND", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)", edge=0.0, decision_details={"market_role": "FAVORITE"})

    if decision_obj.action == "SKIP" and cfg.trade_on_flip:
        trend_edge = decision_obj.edge
        trend_reason = decision_obj.reason
        logger.info(
            "decide_outsider_actual_threshold",
            asset=signal.asset,
            p_flip=round(p_flip, 4),
            flip_threshold_in_local_config=float(local_config.get("FLIP_THRESHOLD", 0.0)),
            resolved_key=_resolved_key,
            base_flip_threshold=round(base_flip_threshold, 4),
            applied_upper=round(upper, 4),
            auto_dead_zone=cfg.auto_dead_zone,
        )
        outsider_obj = decide_outsider(signal, p_flip, local_config, ece=ece)
        if outsider_obj.action == "SKIP":
            final_edge = trend_edge if outsider_obj.edge is None else outsider_obj.edge
            decision_obj = dataclasses.replace(
                outsider_obj,
                reason=f"Тренд: {trend_reason} | Флип: {outsider_obj.reason}",
                edge=final_edge,
            )
            if not decision_obj.decision_details:
                decision_obj = dataclasses.replace(decision_obj, decision_details={"market_role": "OUTSIDER"})
            else:
                decision_obj.decision_details["market_role"] = "OUTSIDER"
        else:
            decision_obj = outsider_obj

    g3_dead_zone = not (lower <= p_flip < upper)
    g4_no_flip = p_flip < lower
    g7_crypto_confirm = None

    # Confirm Gate
    use_crypto_confirm = getattr(cfg, 'use_crypto_confirm', False)
    
    confirm_model_key = None
    confirm_model_version = None
    confirm_direction = None
    confirm_passed = None
    proposed_action = decision_obj.action
    proposed_price = getattr(decision_obj, "price", fresh_yes_price)
    proposed_amount_usdc = getattr(decision_obj, "amount_usdc", cfg.bet_size)

    if decision_obj.action != "SKIP" and use_crypto_confirm and crypto_predictor:
        from polyflip.constants import ASSET_TO_BINANCE_SYMBOL
        binance_symbol = ASSET_TO_BINANCE_SYMBOL.get(market.asset.upper())
        if not binance_symbol:
            decision_obj = dataclasses.replace(
                decision_obj, action="SKIP", reason=f"Crypto confirm: unsupported asset {market.asset.upper()}"
            )
        else:
            await crypto_predictor.load(db_session, binance_symbol)
            model_interval = crypto_predictor.get_interval(binance_symbol)
            candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
            fr = await _get_funding_rate(db_session, binance_symbol)
            crypto_sig = crypto_predictor.predict(candles, binance_symbol, funding_rate=fr)
            
            confirm_model_key = crypto_sig.model_key
            confirm_model_version = crypto_sig.model_version
            confirm_direction = crypto_sig.direction

            if not crypto_sig.features_ok:
                confirm_passed = False
                decision_obj = dataclasses.replace(
                    decision_obj, action="SKIP", reason="Crypto confirm: features invalid"
                )
            else:
                market_direction = "UP" if decision_obj.action == "BUY_YES" else "DOWN"
                if crypto_sig.direction != market_direction:
                    confirm_passed = False
                    decision_obj = dataclasses.replace(
                        decision_obj, action="SKIP", reason=f"Crypto confirm veto: direction is {crypto_sig.direction} vs market {market_direction}"
                    )
                else:
                    confirm_passed = True
        g7_crypto_confirm = (decision_obj.action != "SKIP")

    _min_edge = float(local_config.get("MIN_EDGE", 0.0))
    g5_min_edge = (decision_obj.edge is not None and decision_obj.edge >= _min_edge)
    g6_price_range = not (decision_obj.action == "SKIP" and "price" in (decision_obj.reason or "").lower())

    await log_funnel(
        db_session,
        market_id=market.market_id,
        asset=market.asset,
        trading_mode="ML",
        execution_mode=execution_mode,
        used_model=used_model,
        p_flip=p_flip,
        edge=decision_obj.edge if decision_obj else None,
        fresh_price=fresh_yes_price,
        threshold_lower=lower,
        threshold_upper=upper,
        min_edge_used=_min_edge,
        g1_model_loaded=True,
        g2_price_fetched=True,
        g3_dead_zone=g3_dead_zone,
        g4_no_flip=g4_no_flip,
        g5_min_edge=g5_min_edge,
        g6_price_range=g6_price_range,
        g7_crypto_confirm=g7_crypto_confirm,
        primary_model_key=used_model,
        primary_model_version=model_ver,
        confirm_model_key=confirm_model_key,
        confirm_model_version=confirm_model_version,
        proposed_action=proposed_action,
        proposed_price=proposed_price,
        proposed_amount_usdc=proposed_amount_usdc,
        confirm_direction=confirm_direction,
        confirm_passed=confirm_passed,
        final_action=decision_obj.action if decision_obj else "SKIP",
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None,
    )

    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=p_flip,
        model_ver=model_ver,
        edge=decision_obj.edge if decision_obj else None,
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None,
        used_model_key=used_model,
        confirm_model_key=confirm_model_key,
        confirm_model_version=confirm_model_version,
        applied_lower=lower,
        applied_upper=upper,
    )

async def decide_crypto_mode(
    db_session: AsyncSession,
    api_client: Any,
    market: LiveMarket,
    cfg: TradingConfig,
    raw_settings: dict,
    crypto_predictor: Any,
    start_time: datetime,
    time_left_sec: float,
    models_cache: Optional[Any] = None,
) -> DecisionResult:
    raw_settings = dict(raw_settings)
    _raw_ft = raw_settings.get("FLIP_THRESHOLD")
    if _raw_ft is not None:
        try:
            _ft_val = float(_raw_ft)
            if _ft_val > 1.0:
                raw_settings["FLIP_THRESHOLD"] = str(_ft_val / 100.0)
        except ValueError:
            pass

    from polyflip.constants import ASSET_TO_BINANCE_SYMBOL
    binance_symbol = ASSET_TO_BINANCE_SYMBOL.get(market.asset.upper())
    if not binance_symbol:
        return DecisionResult(None, 0.0, None, None, f"Unsupported crypto asset: {market.asset.upper()}")
    
    await crypto_predictor.load(db_session, binance_symbol)
    
    model_interval = crypto_predictor.get_interval(binance_symbol)
    candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
    fr = await _get_funding_rate(db_session, binance_symbol)
    crypto_sig = crypto_predictor.predict(candles, binance_symbol, funding_rate=fr)
    
    if not crypto_sig.features_ok:
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, "Invalid crypto features", used_model_key=crypto_sig.model_key)

    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
        error_detail = fresh_yes_prices.get("error", "unknown") if fresh_yes_prices else "None returned"
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, f"No fresh yes price: {error_detail}", used_model_key=crypto_sig.model_key)
        
    fresh_yes_price = fresh_yes_prices["current_yes_price"]
    yes_ask = fresh_yes_prices.get("best_ask")
    if yes_ask is None and crypto_sig.direction == "UP":
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, "No YES ask price available", used_model_key=crypto_sig.model_key)

    p_flip_ml = None
    if models_cache and getattr(models_cache, "models", None) and market.asset.upper() in models_cache.models:
        try:
            model = models_cache.models.get(market.asset.upper())
            active_features = models_cache.features.get(market.asset.upper(), [])
            if model and active_features:
                p_flip_ml = await infer_flip_for_market(
                    db_session=db_session,
                    market=market,
                    model=model,
                    active_features=active_features,
                    fresh_price=fresh_yes_price,
                    fresh_spread=market.current_spread or 0.01,
                    start_time=start_time,
                    time_left_sec=time_left_sec,
                    max_time_left=cfg.max_time_left,
                )
        except Exception as e:
            logger.error("crypto_mode_ml_pflip_error", asset=market.asset, error=str(e))
            return DecisionResult(None, 0.0, crypto_sig.model_version, None, "flip inference failed", used_model_key=crypto_sig.model_key)

    no_ask = None
    if crypto_sig.direction == "DOWN":
        if getattr(market, "no_token_id", None):
            no_prices = await api_client.get_market_prices(market.no_token_id)
            if no_prices and no_prices.get("best_ask") is not None:
                no_ask = float(no_prices["best_ask"])
        if no_ask is None:
            return DecisionResult(None, 0.0, crypto_sig.model_version, None, "No NO ask price available", used_model_key=crypto_sig.model_key)

    decision_obj = decide_crypto_trend(crypto_sig, yes_ask or fresh_yes_price, market.volume_5min or 0.0, raw_settings, no_ask=no_ask, p_flip_ml=p_flip_ml)
    if decision_obj.action == "SKIP" and not decision_obj.decision_details:
        decision_obj = dataclasses.replace(decision_obj, decision_details={"market_role": "OUTSIDER" if (yes_ask or fresh_yes_price) < 0.50 else "FAVORITE"})
    if not cfg.trade_on_favorite:
        decision_obj = dataclasses.replace(decision_obj, action="SKIP", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)")
    
    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=0.0,
        model_ver=crypto_sig.model_version if crypto_sig else None,
        edge=decision_obj.edge if decision_obj else None,
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None,
        used_model_key=crypto_sig.model_key,
    )

async def _fetch_lgbm_signal(
    crypto_predictor: Any,
    binance_symbol: str,
    asset_upper: str,
) -> Any:
    """Изолированная LGBM-ветка для asyncio.gather."""
    from polyflip.trading.combined_voting import CryptoSignalProxy
    from polyflip.crypto.predictor import MIN_CANDLES_REQUIRED
    from polyflip.crypto.candle_repository import get_recent_candles
    from polyflip.db.connection import async_session
    
    try:
        async with async_session() as db_session:
            await crypto_predictor.load(db_session, binance_symbol)
            interval = crypto_predictor.get_interval(binance_symbol)
            candles = await get_recent_candles(db_session, binance_symbol, interval=interval, limit=MIN_CANDLES_REQUIRED)
            fr = await _get_funding_rate(db_session, binance_symbol)
            return crypto_predictor.predict(candles, binance_symbol, funding_rate=fr)
    except Exception as exc:
        logger.error("combined_lgbm_error_fallback", asset=asset_upper, error=str(exc))
        from polyflip.crypto.predictor import CryptoSignal
        return CryptoSignal(
            symbol=binance_symbol, p_up=0.0, p_down=0.0,
            direction="NONE", signal_strength=0.0, strike=0.0,
            threshold_up=0.0, threshold_down=0.0, model_version=0,
            features_ok=False, risk_vetoed=False
        )

async def decide_combined_mode(
    db_session: AsyncSession,
    api_client: Any,
    market: LiveMarket,
    cfg: TradingConfig,
    raw_settings: dict,
    models_cache: Any,
    crypto_predictor: Any,
    start_time: datetime,
    time_left_sec: float,
    existing_skipped: Any = None,
    execution_mode: str = "PAPER",
) -> DecisionResult:
    """
    COMBINED-режим:
    1. LightGBM определяет тренд (UP/DOWN) и валидируется через features_ok & risk_vetoed.
    2. LogReg (с фазовым fallback) вычисляет p_flip и net edge для стороны кандидата.
    3. evaluate_combined_entry принимает финальное решение и рассчитывает bet_size.
    4. Записывается ровно ОДИН лог воронки в DecisionFunnelLog со всеми полями.
    """
    import uuid
    import json
    import time
    from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS, COMBINED_BINANCE_SYMBOLS
    from polyflip.trading.combined_voting import evaluate_combined_entry
    from polyflip.crypto.predictor import CryptoSignal
    from polyflip.constants import get_price_phase

    asset_upper = market.asset.upper()

    # Guard: если актив не входит в список поддерживаемых COMBINED активов
    if asset_upper not in COMBINED_MODE_SUPPORTED_ASSETS:
        if execution_mode != "PAPER":
            logger.warning(
                "combined_mode_unsupported_asset_skip",
                asset=asset_upper,
                supported=list(COMBINED_MODE_SUPPORTED_ASSETS),
                execution_mode=execution_mode,
            )
            return DecisionResult(
                decision_obj=TradeDecision(
                    action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
                    reason=f"Asset {asset_upper} not supported in COMBINED mode",
                    strategy_type="COMBINED", p_flip=None, edge=0.0
                ),
                p_flip=0.0,
                model_ver=None,
                edge=None,
                skip_reason=f"Asset {asset_upper} not supported in COMBINED mode",
            )
        else:
            logger.warning(
                "combined_mode_unsupported_asset_fallback_to_ml",
                asset=asset_upper,
                supported=list(COMBINED_MODE_SUPPORTED_ASSETS),
            )
            return await decide_ml_mode(
                db_session, api_client, market, cfg,
                raw_settings, models_cache, None,
                start_time, time_left_sec, existing_skipped,
                execution_mode=execution_mode,
            )

    t0 = time.monotonic()
    decision_run_id = f"dec_{uuid.uuid4().hex[:12]}"

    # 1. Запрашиваем актуальные цены Polymarket (YES и NO)
    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or fresh_yes_prices.get("current_yes_price") is None:
        logger.warning("combined_fresh_prices_failed", asset=asset_upper, market_id=market.market_id)
        await log_funnel(
            db_session,
            market_id=market.market_id,
            asset=market.asset,
            trading_mode="COMBINED",
            execution_mode=execution_mode,
            decision_run_id=decision_run_id,
            p_flip=None,
            edge=None,
            fresh_price=None,
            threshold_lower=cfg.no_flip_threshold,
            threshold_upper=cfg.flip_threshold,
            min_edge_used=cfg.combined_min_net_edge,
            g1_model_loaded=bool(models_cache and models_cache.models),
            g2_price_fetched=False,
            final_action="SKIP",
            skip_reason="Failed to fetch fresh Polymarket YES price",
        )
        return DecisionResult(
            decision_obj=TradeDecision(
                action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
                reason="Failed to fetch fresh Polymarket YES price",
                strategy_type="COMBINED", p_flip=None, edge=0.0
            ),
            p_flip=0.0,
            model_ver=None,
            edge=None,
            skip_reason="Failed to fetch fresh Polymarket YES price",
        )

    fresh_yes_price = float(fresh_yes_prices["current_yes_price"])
    fresh_spread = float(fresh_yes_prices.get("current_spread", market.current_spread or 0.01))
    yes_best_ask = float(fresh_yes_prices["best_ask"]) if fresh_yes_prices.get("best_ask") is not None else fresh_yes_price

    # NO prices
    fresh_no_price = 1.0 - fresh_yes_price
    no_best_ask = 1.0 - fresh_yes_price
    if market.no_token_id:
        fresh_no_prices = await api_client.get_market_prices(market.no_token_id)
        if fresh_no_prices and fresh_no_prices.get("best_ask") is not None:
            no_best_ask = float(fresh_no_prices["best_ask"])
            if fresh_no_prices.get("current_yes_price") is not None:
                fresh_no_price = float(fresh_no_prices["current_yes_price"])

    # 2. Получаем сигнал LightGBM (Direction Model)
    binance_symbol = COMBINED_BINANCE_SYMBOLS.get(asset_upper)
    if crypto_predictor is not None and binance_symbol is not None:
        direction_signal = await _fetch_lgbm_signal(crypto_predictor, binance_symbol, asset_upper)
    else:
        direction_signal = CryptoSignal(
            symbol=binance_symbol or "UNKNOWN", p_up=0.0, p_down=0.0,
            direction="NONE", signal_strength=0.0, strike=0.0,
            threshold_up=0.0, threshold_down=0.0, model_version=0,
            features_ok=False, risk_vetoed=False,
            regime="UNKNOWN", status="PREDICTOR_NOT_AVAILABLE"
        )
        logger.warning("combined_no_crypto_predictor", asset=asset_upper)

    # 3. Выбор LogReg модели (Entry Model) с фазовым fallback
    phase = get_price_phase(fresh_yes_price)
    phase_asset = f"{asset_upper}_{phase}"

    entry_model = None
    entry_model_key = None
    entry_model_ver = None
    entry_features = []
    entry_source = "NONE"
    fallback_reason = None

    if models_cache and models_cache.models:
        if phase_asset in models_cache.models:
            entry_model = models_cache.models[phase_asset]
            entry_model_key = phase_asset
            entry_model_ver = models_cache.versions.get(phase_asset)
            entry_features = models_cache.features.get(phase_asset, [])
            entry_source = "PHASE"
        else:
            if execution_mode == "PAPER":
                if asset_upper in models_cache.models:
                    entry_model = models_cache.models[asset_upper]
                    entry_model_key = asset_upper
                    entry_model_ver = models_cache.versions.get(asset_upper)
                    entry_features = models_cache.features.get(asset_upper, [])
                    entry_source = "BASE"
                    fallback_reason = f"Phase model {phase_asset} not found, fell back to base {asset_upper}"
                elif "GLOBAL" in models_cache.models:
                    entry_model = models_cache.models["GLOBAL"]
                    entry_model_key = "GLOBAL"
                    entry_model_ver = models_cache.versions.get("GLOBAL")
                    entry_features = models_cache.features.get("GLOBAL", [])
                    entry_source = "GLOBAL"
                    fallback_reason = f"Base model {asset_upper} not found, fell back to GLOBAL"
                else:
                    fallback_reason = "No active model matches phase, base asset, or GLOBAL"
            else:
                fallback_reason = f"Phase model {phase_asset} not found, and fallback is forbidden in {execution_mode}"
    else:
        fallback_reason = "ModelsCache is empty"

    # 4. Вычисляем p_flip через Entry Model (если доступна)
    p_flip: Optional[float] = None
    if entry_model is not None:
        try:
            p_flip = await infer_flip_for_market(
                db_session=db_session,
                market=market,
                model=entry_model,
                active_features=entry_features,
                fresh_price=fresh_yes_price,
                fresh_spread=fresh_spread,
                start_time=start_time,
                time_left_sec=time_left_sec,
                max_time_left=cfg.max_time_left,
            )
        except Exception as e:
            logger.error("combined_mode_infer_flip_error", asset=asset_upper, error=str(e))
            p_flip = None
            fallback_reason = f"Infer flip exception: {e}"

    # 5. Оценка входа через evaluate_combined_entry
    comb_cost_buffer = cfg.combined_cost_buffer
    comb_min_net_edge = cfg.combined_min_net_edge
    
    if raw_settings.get("COMBINED_COST_BUFFER") is not None:
        try:
            comb_cost_buffer = float(raw_settings["COMBINED_COST_BUFFER"])
        except (ValueError, TypeError):
            pass
    if raw_settings.get("COMBINED_MIN_NET_EDGE") is not None:
        try:
            comb_min_net_edge = float(raw_settings["COMBINED_MIN_NET_EDGE"])
        except (ValueError, TypeError):
            pass

    vol_5m = 0.0
    try:
        if getattr(market, "volume_5min", None) is not None:
            vol_5m = float(market.volume_5min)
    except (TypeError, ValueError):
        vol_5m = 0.0

    und_price = None
    try:
        if getattr(market, "underlying_price", None) is not None:
            und_price = float(market.underlying_price)
    except (TypeError, ValueError):
        und_price = None

    comb_res = evaluate_combined_entry(
        crypto_sig=direction_signal,
        market_phase=phase,
        entry_requested_key=phase_asset,
        entry_model_key=entry_model_key,
        entry_model_version=entry_model_ver,
        entry_model_source=entry_source,
        p_flip=p_flip,
        fresh_yes_price=fresh_yes_price,
        yes_ask=yes_best_ask,
        no_ask=no_best_ask,
        cost_buffer=comb_cost_buffer,
        min_net_edge=comb_min_net_edge,
        min_price=cfg.trade_min_price,
        max_price=cfg.trade_max_price,
        volume_5min=vol_5m,
        config_dict=raw_settings,
        underlying_price=und_price,
        fallback_reason=fallback_reason,
    )

    elapsed = time.monotonic() - t0
    logger.info("combined_mode_latency", asset=asset_upper, elapsed_ms=round(elapsed * 1000, 1))

    # 6. Формируем decision_details и TradeDecision
    decision_details = {
        "decision_run_id": decision_run_id,
        "direction_status": comb_res.direction_status,
        "direction_model_key": comb_res.direction_model_key,
        "direction_model_version": comb_res.direction_model_version,
        "direction_regime": comb_res.direction_regime,
        "direction_probability": comb_res.direction_probability,
        "direction_value": comb_res.direction_value,
        "entry_requested_key": phase_asset,
        "entry_model_key": comb_res.entry_model_key,
        "entry_model_version": comb_res.entry_model_version,
        "entry_model_phase": comb_res.entry_model_phase,
        "entry_model_source": comb_res.entry_model_source,
        "entry_status": comb_res.entry_status,
        "fallback_reason": comb_res.fallback_reason,
        "p_candidate_win": comb_res.p_candidate_win,
        "candidate_side": comb_res.candidate_side,
        "candidate_ask": comb_res.candidate_ask,
        "gross_edge": comb_res.gross_edge,
        "cost_buffer": comb_res.cost_buffer,
        "net_edge": comb_res.net_edge,
        "max_acceptable_price": comb_res.max_acceptable_price,
        "strike_source": comb_res.strike_source,
        "strike_proxy": comb_res.strike_proxy,
        "underlying_price": comb_res.underlying_price,
        "distance_to_strike_pct": comb_res.distance_to_strike_pct,
        "market_role": "FAVORITE" if (comb_res.candidate_ask is not None and comb_res.candidate_ask >= 0.50) else "OUTSIDER",
    }

    lgbm_meta_dict = {
        "lgbm_version": comb_res.direction_model_version,
        "lgbm_model_key": comb_res.direction_model_key,
        "lgbm_direction": comb_res.direction_value,
        "lgbm_features_ok": (comb_res.direction_status == "READY"),
        "is_fallback": (comb_res.entry_model_source in ("BASE", "GLOBAL")),
        "vote_action": comb_res.action,
        "bet_size_multiplier": 1.0,
        "trading_mode": "COMBINED",
        "ml_phase_model": comb_res.entry_model_key,
        "original_strategy": "COMBINED",
    }
    lgbm_meta = json.dumps(lgbm_meta_dict)

    if comb_res.action != "SKIP":
        trade_decision = TradeDecision(
            action=comb_res.action,
            buy_price=comb_res.candidate_ask or 0.0,
            bet_size_usdc=comb_res.bet_size_usdc,
            reason=comb_res.reason,
            strategy_type="COMBINED",
            p_flip=comb_res.p_flip,
            p_up=comb_res.direction_probability if comb_res.direction_value == "UP" else (1.0 - comb_res.direction_probability if comb_res.direction_probability is not None else None),
            strike=comb_res.strike_proxy,
            edge=comb_res.net_edge,
            p_win_effective=comb_res.p_candidate_win,
            p_win_raw=comb_res.p_candidate_win,
            decision_details=decision_details,
        )
    else:
        trade_decision = TradeDecision(
            action="SKIP",
            buy_price=comb_res.candidate_ask or 0.0,
            bet_size_usdc=0.0,
            reason=comb_res.reason,
            strategy_type="COMBINED",
            p_flip=comb_res.p_flip,
            p_up=comb_res.direction_probability if comb_res.direction_value == "UP" else None,
            strike=comb_res.strike_proxy,
            edge=comb_res.net_edge or 0.0,
            decision_details=decision_details,
        )

    # 7. Записываем воронку (DecisionFunnelLog) — ровно одна запись!
    g1_loaded = (entry_model is not None and comb_res.direction_status != "MODEL_NOT_LOADED")
    g8_vote = (comb_res.action != "SKIP")

    await log_funnel(
        db_session,
        market_id=market.market_id,
        asset=market.asset,
        trading_mode="COMBINED",
        execution_mode=execution_mode,
        decision_run_id=decision_run_id,
        used_model=comb_res.entry_model_key,
        p_flip=comb_res.p_flip,
        edge=comb_res.net_edge,
        fresh_price=comb_res.candidate_ask or fresh_yes_price,
        threshold_lower=cfg.no_flip_threshold,
        threshold_upper=cfg.flip_threshold,
        min_edge_used=comb_min_net_edge,
        g1_model_loaded=g1_loaded,
        g2_price_fetched=True,
        g8_combined_vote=g8_vote,
        primary_model_key=comb_res.entry_model_key,
        primary_model_version=comb_res.entry_model_version,
        confirm_model_key=comb_res.direction_model_key,
        confirm_model_version=comb_res.direction_model_version,
        proposed_action=comb_res.action,
        proposed_price=comb_res.candidate_ask,
        proposed_amount_usdc=comb_res.bet_size_usdc if comb_res.action != "SKIP" else 0.0,
        confirm_direction=comb_res.direction_value,
        confirm_passed=(comb_res.direction_status == "READY"),
        
        # Новая телеметрия
        direction_status=comb_res.direction_status,
        direction_model_key=comb_res.direction_model_key,
        direction_model_version=comb_res.direction_model_version,
        direction_regime=comb_res.direction_regime,
        direction_probability=comb_res.direction_probability,
        direction_p_up=comb_res.direction_p_up,
        direction_p_down=comb_res.direction_p_down,
        required_direction_model_key=f"{asset_upper}_{comb_res.direction_regime}" if comb_res.direction_regime else None,
        direction_value=comb_res.direction_value,
        entry_requested_key=phase_asset,
        entry_model_key=comb_res.entry_model_key,
        entry_model_version=comb_res.entry_model_version,
        entry_model_phase=comb_res.entry_model_phase,
        entry_model_source=comb_res.entry_model_source,
        entry_status=comb_res.entry_status,
        fallback_reason=comb_res.fallback_reason,
        p_candidate_win=comb_res.p_candidate_win,
        candidate_side=comb_res.candidate_side,
        candidate_ask=comb_res.candidate_ask,
        gross_edge=comb_res.gross_edge,
        cost_buffer=comb_res.cost_buffer,
        net_edge=comb_res.net_edge,
        max_acceptable_price=comb_res.max_acceptable_price,
        strike_source=comb_res.strike_source,
        strike_proxy=comb_res.strike_proxy,
        underlying_price=comb_res.underlying_price,
        distance_to_strike_pct=comb_res.distance_to_strike_pct,

        final_action=comb_res.action,
        skip_reason=comb_res.reason if comb_res.action == "SKIP" else None,
    )

    return DecisionResult(
        decision_obj=trade_decision,
        p_flip=comb_res.p_flip if comb_res.p_flip is not None else 0.0,
        model_ver=comb_res.entry_model_version,
        edge=comb_res.net_edge,
        skip_reason=comb_res.reason if comb_res.action == "SKIP" else None,
        lgbm_metadata=lgbm_meta,
        used_model_key=comb_res.entry_model_key,
        confirm_model_key=comb_res.direction_model_key,
        confirm_model_version=comb_res.direction_model_version,
    )
