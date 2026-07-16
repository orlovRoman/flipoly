import dataclasses
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision, MarketSignal, decide_favorite, decide_crypto_trend, decide_ml_trend, decide_outsider
from polyflip.trading.ml_inference import build_inference_dataframe, run_model_inference
from polyflip.crypto.predictor import MIN_CANDLES_REQUIRED
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.trading.utils import compute_dead_zone

logger = structlog.get_logger(__name__)

def _get_float_setting(raw_settings: dict, key: str) -> Optional[float]:
    val = raw_settings.get(key)
    if val is not None and str(val).strip() not in ("", "0", "0.0"):
        try:
            return float(val)
        except ValueError:
            pass
    return None

@dataclass
class DecisionResult:
    decision_obj: Optional[TradeDecision]
    p_flip: float
    model_ver: Optional[int]
    edge: Optional[float]
    skip_reason: Optional[str]

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
        time_left_min=time_left_sec / 60.0
    )
    
    local_fav_config = {
        "FAVORITE_THRESHOLD": str(cfg.favorite_threshold),
        "FAVORITE_MIN_EDGE": str(cfg.favorite_min_edge),
        "DEAD_ZONE_WIDTH": str(cfg.dead_zone),
        "FAVORITE_MIN_PRICE": str(cfg.favorite_min_price),
        "FAVORITE_MAX_PRICE": str(cfg.favorite_max_price),
        "TRADE_BET_SIZE_USDC": str(cfg.bet_size),
        "MAX_BET_SIZE_USDC": str(cfg.max_bet_size_usdc),
        "MAX_BET_EDGE": str(cfg.max_bet_edge),
        "MIN_EDGE": str(asset_min_edge),
        "TRADE_MAX_PRICE": str(asset_max_price),
        "LIQUIDITY_FRACTION": str(cfg.liquidity_fraction),
        "BET_SIZING_MODE": str(cfg.bet_sizing_mode),
    }
    
    decision_obj = decide_favorite(signal, local_fav_config)
    if not cfg.trade_on_favorite:
        decision_obj = dataclasses.replace(decision_obj, action="SKIP", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)")
    
    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=0.0,
        model_ver=None,
        edge=decision_obj.edge,
        skip_reason=decision_obj.reason if decision_obj.action == "SKIP" else None
    )

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
) -> DecisionResult:
    if raw_settings is None:
        raw_settings = {}
    if start_time is None:
        start_time = datetime.now(timezone.utc)
    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
        error_msg = fresh_yes_prices.get("error", "No fresh YES prices from API") if fresh_yes_prices else "No fresh YES prices from API"
        return DecisionResult(None, 0.0, None, None, error_msg)

    fresh_yes_price = fresh_yes_prices["current_yes_price"]
    fresh_spread = fresh_yes_prices.get("current_spread", market.current_spread)
    
    from polyflip.constants import get_price_phase
    phase = get_price_phase(fresh_yes_price)
    phase_asset = f"{market.asset.upper()}_{phase}"
    
    from polyflip.trading.ml_inference import populate_models_cache
    await populate_models_cache(db_session)

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
    
    if not model:
        return DecisionResult(None, 0.0, None, None, f"No active model found for {market.asset.upper()}")
        
    df = build_inference_dataframe(
        market=market,
        history_snaps=[], 
        fresh_yes_price=fresh_yes_price,
        fresh_spread=fresh_spread,
        global_max=market.current_yes_price,
        start_time=start_time,
        time_left_sec=time_left_sec,
    )
    
    p_flip = float(run_model_inference(df, model, active_features))
    
    signal = MarketSignal(
        asset=market.asset,
        mid_price=fresh_yes_price,
        spread=fresh_spread or 0.01,
        volume_5min=market.volume_5min or 0.0,
        price_velocity=market.price_velocity or 0.0,
        hour_of_day=start_time.hour,
        time_left_min=time_left_sec / 60.0
    )

    # Приоритет: ручной → фазовый авто → базовый авто → дефолт
    manual_key     = f"TRADE_FLIP_THRESHOLD_{market.asset.upper()}"
    auto_key_phase = f"AUTO_FLIP_THRESHOLD_{used_model}"
    auto_key_base  = f"AUTO_FLIP_THRESHOLD_{market.asset.upper()}"

    manual_val = _get_float_setting(raw_settings, manual_key)
    auto_phase_val = _get_float_setting(raw_settings, auto_key_phase)
    auto_base_val = _get_float_setting(raw_settings, auto_key_base)
    global_threshold_val = _get_float_setting(raw_settings, "TRADE_FLIP_THRESHOLD")

    if manual_val is not None:
        base_flip_threshold = manual_val
    elif auto_phase_val is not None:
        base_flip_threshold = auto_phase_val
    elif auto_base_val is not None:
        base_flip_threshold = auto_base_val
    elif global_threshold_val is not None:
        base_flip_threshold = global_threshold_val
    else:
        base_flip_threshold = cfg.no_flip_threshold + cfg.dead_zone

    lower, upper = compute_dead_zone(
        flip_threshold=base_flip_threshold,
        dead_zone_width=cfg.dead_zone,
        auto_mode=cfg.auto_dead_zone,
    )

    local_config = {**raw_settings}
    local_config["NO_FLIP_THRESHOLD"] = str(lower)
    local_config["FLIP_THRESHOLD"] = str(upper)
    # Используем FAVORITE_MIN_EDGE=-100.0 для обхода edge-фильтрации внутри вспомогательного вызова decide_favorite.
    # Это сохраняет оригинальный MIN_EDGE в local_config для правильной ML-фильтрации в decide_ml_trend и расчета Kelly.
    local_config["FAVORITE_MIN_EDGE"] = "-100.0"
    local_config["MAX_BET_EDGE"] = "100.0"
    local_config["BYPASS_BET_SIZE_CHECK"] = "true"

    if cfg.trade_on_favorite:
        decision_obj = decide_ml_trend(signal, p_flip, local_config)
    else:
        decision_obj = TradeDecision(action="SKIP", buy_price=0.0, bet_size_usdc=0.0, strategy_type="ML_TREND", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)", edge=0.0)

    if decision_obj.action == "SKIP" and cfg.trade_on_flip:
        decision_obj = decide_outsider(signal, p_flip, local_config)

    if decision_obj.action == "SKIP" and decision_obj.reason != "Favorite trades disabled (TRADE_ON_FAVORITE=False)":
        if lower <= p_flip < upper:
            decision_obj = dataclasses.replace(
                decision_obj, reason=f"Мёртвая зона (p_flip={p_flip:.2f} в [{lower:.2f}, {upper:.2f}])"
            )
        elif not cfg.trade_on_flip and p_flip >= upper:
            decision_obj = dataclasses.replace(
                decision_obj, reason=f"Ожидается флип (p_flip={p_flip:.2f} >= {upper:.2f})"
            )

    # Confirm Gate
    use_crypto_confirm = getattr(cfg, 'use_crypto_confirm', False)
    if decision_obj.action != "SKIP" and use_crypto_confirm and crypto_predictor:
        binance_symbol = "BTCUSDT" if market.asset.upper() == "BTC" else "ETHUSDT"
        await crypto_predictor.load(db_session, binance_symbol)
        
        model_interval = crypto_predictor.get_interval(binance_symbol)
        candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
        crypto_sig = crypto_predictor.predict(candles, binance_symbol)
        
        if not crypto_sig.features_ok:
            decision_obj = dataclasses.replace(
                decision_obj, action="SKIP", reason="Crypto confirm: features invalid"
            )
        else:
            market_direction = "UP" if decision_obj.action == "BUY_YES" else "DOWN"
            if crypto_sig.direction != market_direction:
                decision_obj = dataclasses.replace(
                    decision_obj, action="SKIP", reason=f"Crypto confirm veto: direction is {crypto_sig.direction} vs market {market_direction}"
                )

    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=p_flip,
        model_ver=model_ver,
        edge=decision_obj.edge if decision_obj else None,
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None
    )

async def decide_crypto_mode(
    db_session: AsyncSession,
    api_client: Any,
    market: LiveMarket,
    cfg: TradingConfig,
    raw_settings: dict,
    crypto_predictor: Any,
    start_time: datetime,
    time_left_sec: float
) -> DecisionResult:
    binance_symbol = "BTCUSDT" if market.asset.upper() == "BTC" else "ETHUSDT"
    
    await crypto_predictor.load(db_session, binance_symbol)
    
    model_interval = crypto_predictor.get_interval(binance_symbol)
    candles = await get_recent_candles(db_session, binance_symbol, interval=model_interval, limit=MIN_CANDLES_REQUIRED)
    crypto_sig = crypto_predictor.predict(candles, binance_symbol)
    
    if not crypto_sig.features_ok:
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, "Invalid crypto features")

    fresh_yes_prices = await api_client.get_market_prices(market.yes_token_id)
    if not fresh_yes_prices or "current_yes_price" not in fresh_yes_prices:
        error_detail = fresh_yes_prices.get("error", "unknown") if fresh_yes_prices else "None returned"
        return DecisionResult(None, 0.0, crypto_sig.model_version, None, f"No fresh yes price: {error_detail}")
        
    fresh_yes_price = fresh_yes_prices["current_yes_price"]

    decision_obj = decide_crypto_trend(crypto_sig, fresh_yes_price, market.volume_5min or 0.0, raw_settings)
    if not cfg.trade_on_favorite:
        decision_obj = dataclasses.replace(decision_obj, action="SKIP", reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)")
    
    return DecisionResult(
        decision_obj=decision_obj,
        p_flip=0.0,
        model_ver=crypto_sig.model_version if crypto_sig else None,
        edge=decision_obj.edge if decision_obj else None,
        skip_reason=decision_obj.reason if decision_obj and decision_obj.action == "SKIP" else None
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
) -> DecisionResult:
    """
    COMBINED-режим: ML (LogReg) + LightGBM голосуют, решение по таблице.
    Поддерживается только для активов из COMBINED_MODE_SUPPORTED_ASSETS.
    """
    from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS, COMBINED_BINANCE_SYMBOLS
    from polyflip.trading.combined_voting import combine_votes, CryptoSignalProxy
    from polyflip.crypto.predictor import MIN_CANDLES_REQUIRED
    from polyflip.crypto.candle_repository import get_recent_candles

    asset_upper = market.asset.upper()

    # Guard: если актив не поддерживает LightGBM — деградируем в ML-режим
    if asset_upper not in COMBINED_MODE_SUPPORTED_ASSETS:
        logger.warning(
            "combined_mode_unsupported_asset_fallback_to_ml",
            asset=asset_upper,
            supported=list(COMBINED_MODE_SUPPORTED_ASSETS),
        )
        return await decide_ml_mode(
            db_session, api_client, market, cfg,
            raw_settings, models_cache, None,
            start_time, time_left_sec, existing_skipped,
        )

    # --- Шаг A: запускаем ML-ветку ---
    ml_result = await decide_ml_mode(
        db_session, api_client, market, cfg,
        raw_settings, models_cache, None,   # crypto_predictor=None, вето здесь не нужно
        start_time, time_left_sec, existing_skipped,
    )

    # --- Шаг B: запускаем LightGBM-ветку ---
    binance_symbol = COMBINED_BINANCE_SYMBOLS.get(asset_upper)
    crypto_proxy = CryptoSignalProxy(direction=None, features_ok=False)

    if crypto_predictor is not None and binance_symbol is not None:
        try:
            await crypto_predictor.load(db_session, binance_symbol)
            model_interval = crypto_predictor.get_interval(binance_symbol)
            candles = await get_recent_candles(
                db_session, binance_symbol,
                interval=model_interval, limit=MIN_CANDLES_REQUIRED
            )
            raw_sig = crypto_predictor.predict(candles, binance_symbol)
            crypto_proxy = CryptoSignalProxy(
                direction=raw_sig.direction,
                features_ok=raw_sig.features_ok,
                model_version=getattr(raw_sig, "model_version", None),
            )
        except Exception as exc:
            logger.error("combined_lgbm_error_fallback", asset=asset_upper, error=str(exc))
            # features_ok=False → combine_votes сделает fallback на ML
    else:
        logger.warning("combined_no_crypto_predictor", asset=asset_upper)

    # --- Шаг C: голосование ---
    ml_action = ml_result.decision_obj.action if ml_result.decision_obj else "SKIP"
    ml_edge   = ml_result.edge or 0.0

    vote = combine_votes(ml_action, ml_edge, crypto_proxy, asset_upper)

    logger.info(
        "combined_vote_result",
        asset=asset_upper,
        ml_action=vote.ml_action,
        lgbm_direction=vote.lgbm_direction,
        lgbm_features_ok=vote.lgbm_features_ok,
        final_action=vote.action,
        reason=vote.reason,
        confidence=round(vote.confidence, 4),
    )

    # --- Шаг D: применяем результат голосования к decision_obj ---
    import dataclasses
    if ml_result.decision_obj is None:
        return DecisionResult(None, ml_result.p_flip, ml_result.model_ver, None, vote.reason)

    final_decision = dataclasses.replace(
        ml_result.decision_obj,
        action=vote.action,
        reason=vote.reason,
    )

    return DecisionResult(
        decision_obj=final_decision,
        p_flip=ml_result.p_flip,
        model_ver=ml_result.model_ver,
        edge=final_decision.edge if vote.action != "SKIP" else None,
        skip_reason=vote.reason if vote.action == "SKIP" else None,
    )
