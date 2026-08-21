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
from polyflip.trading.decision_logic import TradeDecision, MarketSignal, decide_outsider
from polyflip.trading.ml_inference import build_inference_dataframe, run_model_inference
from polyflip.crypto.predictor import MIN_CANDLES_REQUIRED
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.trading.utils import compute_dead_zone
from polyflip.trading.funnel_logger import log_funnel
from polyflip.crypto.market_regime_integration import build_snapshot_from_candles
from polyflip.crypto.market_regime_apply import apply_regime_policy
from polyflip.crypto.market_regime_classifier import Regime

logger = structlog.get_logger(__name__)


def _resolve_lgbm_attribution(
    mode: str,
    direction_value: Any,
    model_key: Optional[str],
    model_version: Optional[int],
) -> dict[str, Any]:
    """Normalize LGBM direction telemetry and separate applied vs observed attribution.

    A loaded model is not an applied directional model when it abstains (NONE or an
    empty value).  Shadow mode still exposes the loaded model for telemetry, while
    ``direction_model_key`` in active/funnel attribution only identifies a model that
    selected UP or DOWN.
    """
    if mode == "OFF":
        return {
            "direction_value": "NONE",
            "actually_decided": False,
            "applied_model_key": None,
            "applied_model_version": None,
            "shadow_model_key": None,
            "shadow_model_version": None,
            "funnel_model_key": None,
            "funnel_model_version": None,
        }
    normalized_value = str(direction_value or "").strip().upper()
    if normalized_value not in {"UP", "DOWN"}:
        normalized_value = "NONE"
    actually_decided = normalized_value in {"UP", "DOWN"}
    is_active = mode == "ACTIVE"
    is_shadow = mode == "SHADOW"
    return {
        "direction_value": normalized_value,
        "actually_decided": actually_decided,
        "applied_model_key": model_key if is_active and actually_decided else None,
        "applied_model_version": model_version if is_active and actually_decided else None,
        "shadow_model_key": model_key if is_shadow else None,
        "shadow_model_version": model_version if is_shadow else None,
        "funnel_model_key": model_key if (is_active and actually_decided) or is_shadow else None,
        "funnel_model_version": model_version if (is_active and actually_decided) or is_shadow else None,
    }


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
    from polyflip.constants import resolve_binance_symbol

    closed_candles = []
    candle_symbol = resolve_binance_symbol(market.asset)
    if candle_symbol:
        recent_candles = await get_recent_candles(
            db_session, candle_symbol, "15m", limit=32
        )
        for candle in recent_candles:
            close_time = getattr(candle, "close_time", None)
            if close_time is None or getattr(candle, "is_closed", None) is not True:
                continue
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            if close_time <= start_time:
                closed_candles.append(candle)


    df = build_inference_dataframe(
        market=market,
        history_snaps=list(history_snaps), 
        fresh_yes_price=fresh_price,
        fresh_spread=fresh_spread,
        global_max=global_max,
        start_time=start_time,
        time_left_sec=time_left_sec,
        closed_candles=closed_candles,
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

async def _fetch_lgbm_signal(
    crypto_predictor: Any,
    binance_symbol: str,
    asset_upper: str,
    cfg: Any,
    market: LiveMarket,
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
            candles = await get_recent_candles(
                db_session,
                binance_symbol,
                interval,
                limit=MIN_CANDLES_REQUIRED,
            )
            funding_rate = await _get_funding_rate(db_session, binance_symbol)
            from polyflip.crypto.market_direction_service import get_or_create_market_direction_signal
            return await get_or_create_market_direction_signal(
                db_session,
                market,
                candles,
                crypto_predictor,
                funding_rate=funding_rate,
                invert_lgbm_signal=cfg.invert_lgbm_signal,
            )
    except Exception as exc:
        logger.error("combined_lgbm_error_fallback", asset=asset_upper, error=str(exc))
        from polyflip.crypto.predictor import CryptoSignal
        return CryptoSignal(
            symbol=binance_symbol, p_up=0.0, p_down=0.0,
            direction="NONE", signal_strength=0.0, strike=0.0,

            threshold_up=0.0, threshold_down=0.0, model_version=-1,
            features_ok=False, risk_vetoed=False, regime="UNKNOWN",
            status="INFERENCE_FAILED", risk_reason=str(exc), model_key=""
        )



async def _apply_mrf_filter(
    db_session,
    cfg,
    asset_upper: str,
    binance_symbol: str,
    start_time,
    candidate_side: str | None,
    fresh_yes_price: float,
    bet_size_usdc: float,
    action: str,
    decision_run_id: str = "",
    lgbm_applied: bool = False,
):
    """Apply MRF filter. Returns (adjusted_action, adjusted_bet_size, mrf_audit, outcome)."""
    if cfg.mrf_mode == "OFF":
        return action, bet_size_usdc, None, None

    try:
        import asyncio
        from polyflip.crypto.candle_repository import get_recent_candles
        from polyflip.constants import COMBINED_BINANCE_SYMBOLS, COMBINED_MODE_SUPPORTED_ASSETS
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles

        limit = cfg.mrf_min_history + 10
        tasks = {}
        for asset_name in COMBINED_MODE_SUPPORTED_ASSETS:
            sym = COMBINED_BINANCE_SYMBOLS.get(asset_name)
            if sym:
                tasks[asset_name] = get_recent_candles(
                    db_session, sym, "15m", limit=limit,
                )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        candles_by_asset = {}
        for asset_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning("mrf_candle_fetch_error", asset=asset_name, error=str(result))
                continue
            if result:
                candles_by_asset[asset_name] = result

        if not candles_by_asset:
            logger.warning("mrf_no_candles_any_asset")
            return action, bet_size_usdc, None, None

        snapshot = build_snapshot_from_multi_asset_candles(candles_by_asset, as_of=start_time)

        if not snapshot.basket.history_ready:
            logger.info("mrf_history_not_ready", asset=asset_upper,
                        reason_codes=snapshot.reason_codes)
            return action, bet_size_usdc, None, None

        outcome = apply_regime_policy(
            cfg=cfg,
            snapshot=snapshot,
            candidate_side=candidate_side,
            fresh_yes_price=fresh_yes_price,
            lgbm_applied=lgbm_applied,
            bet_size_usdc=bet_size_usdc,
            action=action,
            decision_run_id=decision_run_id,
        )

        logger.info(
            "mrf_applied",
            asset=asset_upper,
            mode=cfg.mrf_mode,
            global_phase=outcome.global_phase,
            asset_phase=outcome.asset_phase,
            original_action=action,
            adjusted_action=outcome.adjusted_action,
            original_bet=bet_size_usdc,
            adjusted_bet=outcome.adjusted_bet_size,
            applied=outcome.applied,
        )

        return outcome.adjusted_action, outcome.adjusted_bet_size, outcome.audit_dict, outcome

    except Exception as exc:
        logger.error("mrf_error", asset=asset_upper, error=str(exc))
        return action, bet_size_usdc, None, None

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
            threshold_lower=cfg.outsider_max_price,
            threshold_upper=cfg.flip_threshold,
            min_edge_used=cfg.favorite_min_edge,
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
    lgbm_mode = getattr(cfg, "lightgbm_decision_mode", "SHADOW")

    if lgbm_mode in {"ACTIVE", "SHADOW"} and crypto_predictor is not None and binance_symbol is not None:
        direction_signal = await _fetch_lgbm_signal(
            crypto_predictor,
            binance_symbol,
            asset_upper,
            cfg,
            market,
        )
    else:
        direction_signal = CryptoSignal(
            symbol=binance_symbol or "UNKNOWN", p_up=0.0, p_down=0.0,
            direction="NONE", signal_strength=0.0, strike=0.0,
            threshold_up=0.0, threshold_down=0.0, model_version=-1,
            features_ok=False, risk_vetoed=False,
            regime="UNKNOWN", status="DISABLED_BY_OPERATOR" if lgbm_mode == "OFF" else "PREDICTOR_NOT_AVAILABLE"
        )
        if lgbm_mode == "OFF":
            logger.info("combined_lightgbm_disabled_by_operator", asset=asset_upper)
        else:
            logger.warning("combined_no_crypto_predictor", asset=asset_upper)

    # 2.1 Fallback на ML (LogReg) режим, если LightGBM выдал NONE и включена настройка COMBINED_FALLBACK_TO_ML_ON_NONE
    # cfg уже содержит актуальное значение из raw_settings (через parse_trading_settings) — источник один.
    

    # 3. Выбор LogReg модели (Entry Model) с фазовым fallback
    phase = get_price_phase(fresh_yes_price)
    phase_asset = f"{asset_upper}_{phase}"

    entry_model = None
    entry_model_key = None
    entry_model_ver = None
    entry_features = []
    entry_source = "NONE"
    fallback_reason = None
    p_flip: Optional[float] = None

    candidate_specs: list[tuple[str, str]] = []
    fallback_notes: list[str] = []
    if models_cache and models_cache.models:
        if phase_asset in models_cache.models:
            candidate_specs.append((phase_asset, "PHASE"))
        elif execution_mode == "PAPER":
            fallback_notes.append(f"Phase model {phase_asset} not found")
        else:
            fallback_reason = (
                f"Phase model {phase_asset} not found, and fallback is "
                f"forbidden in {execution_mode}"
            )

        if execution_mode == "PAPER":
            if asset_upper in models_cache.models and asset_upper != phase_asset:
                candidate_specs.append((asset_upper, "BASE"))
            elif asset_upper not in models_cache.models:
                fallback_notes.append(f"Base model {asset_upper} not found")
            if "GLOBAL" in models_cache.models:
                candidate_specs.append(("GLOBAL", "GLOBAL"))
    else:
        fallback_reason = "ModelsCache is empty"

    for candidate_key, candidate_source in candidate_specs:
        entry_model = models_cache.models[candidate_key]
        entry_model_key = candidate_key
        entry_model_ver = models_cache.versions.get(candidate_key)
        entry_features = models_cache.features.get(candidate_key, [])
        entry_source = candidate_source
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
                max_time_left=max(cfg.favor_max_time_left, cfg.outs_max_time_left),
            )
            if candidate_source != "PHASE":
                fallback_notes.append(
                    f"fell back to {candidate_source.lower()} {candidate_key}"
                )
                fallback_reason = "; ".join(fallback_notes)
            break
        except Exception as exc:
            logger.error(
                "combined_mode_infer_flip_error",
                asset=asset_upper,
                model_key=candidate_key,
                model_source=candidate_source,
                error=str(exc),
            )
            p_flip = None
            fallback_notes.append(
                f"{candidate_source} model {candidate_key} inference failed: {exc}"
            )
            if execution_mode != "PAPER":
                break
    else:
        if fallback_notes:
            fallback_reason = "; ".join(fallback_notes)
        elif fallback_reason is None:
            fallback_reason = "No active model matches phase, base asset, or GLOBAL"

    if p_flip is None and fallback_notes:
        fallback_reason = "; ".join(fallback_notes)

    # 5. Оценка входа через evaluate_combined_entry
    comb_cost_buffer = cfg.combined_cost_buffer
    
    if raw_settings.get("COMBINED_COST_BUFFER") is not None:
        try:
            comb_cost_buffer = float(raw_settings["COMBINED_COST_BUFFER"])
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
    entry_model_ece = 0.0
    if models_cache and entry_model_key and hasattr(models_cache, "eces"):
        ece_map = getattr(models_cache, "eces", None)
        if isinstance(ece_map, dict):
            val = ece_map.get(entry_model_key, 0.0)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                entry_model_ece = float(val)

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
        cfg=cfg,
        volume_5min=vol_5m,
        underlying_price=und_price,
        time_left_sec=time_left_sec,
        fallback_reason=fallback_reason,
        entry_model_ece=entry_model_ece,
    )

    elapsed = time.monotonic() - t0
    logger.info("combined_mode_latency", asset=asset_upper, elapsed_ms=round(elapsed * 1000, 1))

    lgbm_mode = getattr(cfg, "lightgbm_decision_mode", "SHADOW")
    lgbm_applied = (lgbm_mode == "ACTIVE")
    lgbm_shadow = (lgbm_mode == "SHADOW")
    lgbm_attribution = _resolve_lgbm_attribution(
        lgbm_mode,
        comb_res.direction_value,
        comb_res.direction_model_key,
        comb_res.direction_model_version,
    )
    lgbm_direction_value = lgbm_attribution["direction_value"]
    applied_direction_key = lgbm_attribution["applied_model_key"]
    applied_direction_version = lgbm_attribution["applied_model_version"]

    # 6. Формируем decision_details и TradeDecision
    decision_details = {
        "decision_run_id": decision_run_id,
        "lightgbm_decision_mode": lgbm_mode,
        "lightgbm_applied": lgbm_applied,
        "decision_source": "LOGREG_PLUS_LIGHTGBM" if lgbm_applied else "LOGREG_ONLY",
        "consensus_type": comb_res.consensus_type,
        "direction_status": "SHADOW_NOT_APPLIED" if lgbm_shadow else ("DISABLED_BY_OPERATOR" if lgbm_mode == "OFF" else comb_res.direction_status),
        "direction_model_key": applied_direction_key,
        "direction_model_version": applied_direction_version,
        "shadow_direction_model_key": lgbm_attribution["shadow_model_key"],
        "shadow_direction_model_version": lgbm_attribution["shadow_model_version"],
        "shadow_direction_value": lgbm_direction_value if lgbm_shadow else None,
        "shadow_direction_probability": comb_res.direction_probability if lgbm_shadow else None,
        "shadow_inference_status": direction_signal.status if direction_signal else None,
        "shadow_features_ok": direction_signal.features_ok if direction_signal else False,
        "direction_regime": comb_res.direction_regime,
        "direction_probability": comb_res.direction_probability,
        "direction_value": lgbm_direction_value,
        "raw_opinion": comb_res.direction_raw_opinion,
        "actionable_signal": lgbm_direction_value,
        "direction_p_up_raw": comb_res.direction_p_up_raw,
        "direction_p_down_raw": comb_res.direction_p_down_raw,
        "entry_requested_key": phase_asset,
        "entry_model_key": comb_res.entry_model_key,
        "entry_model_version": comb_res.entry_model_version,
        "entry_model_phase": comb_res.entry_model_phase,
        "entry_model_source": comb_res.entry_model_source,
        "entry_status": comb_res.entry_status,
        "fallback_reason": comb_res.fallback_reason,
        "p_candidate_win": comb_res.p_candidate_win,
        "p_logreg_win": comb_res.p_logreg_win,
        "direction_discount_applied": comb_res.direction_discount_applied,
        "combined_dir_discount_weight": comb_res.combined_dir_discount_weight,
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
        "would_live_accept": comb_res.would_live_accept,
        "p_flip_raw": comb_res.p_flip_raw,
        "p_flip_effective": comb_res.p_flip_effective,
        "entry_model_ece": comb_res.entry_model_ece,
        "market_role": "FAVORITE" if (comb_res.candidate_ask is not None and comb_res.candidate_ask >= 0.50) else "OUTSIDER",
    }

    if comb_res.lgbm_inverted:
        decision_details["lgbm_signal"] = "[INVERTED]"
        decision_details["lgbm_p_up_raw"] = comb_res.lgbm_p_up_raw
        decision_details["lgbm_p_down_raw"] = comb_res.lgbm_p_down_raw
        decision_details["lgbm_p_up_used"] = 1.0 - comb_res.lgbm_p_up_raw
        decision_details["lgbm_p_down_used"] = 1.0 - comb_res.lgbm_p_down_raw
    else:
        decision_details["lgbm_signal"] = "normal"

    lgbm_meta_dict = {
        # SHADOW retains model attribution even though it is excluded from trading.
        "lgbm_version": comb_res.direction_model_version,
        "lgbm_model_key": comb_res.direction_model_key,
        "lgbm_direction": lgbm_direction_value,
        "lgbm_features_ok": direction_signal.features_ok if direction_signal else False,
        "shadow_inference_status": direction_signal.status if direction_signal else "NONE",
        "lgbm_regime": comb_res.direction_regime,
        "lgbm_probability": comb_res.direction_probability,
        "lgbm_p_up": comb_res.direction_p_up,
        "lgbm_p_down": comb_res.direction_p_down,
        "lightgbm_observed": lgbm_mode in {"ACTIVE", "SHADOW"},
        "is_fallback": (comb_res.entry_model_source in ("BASE", "GLOBAL")),
        "vote_action": comb_res.action,
        "bet_size_multiplier": 1.0,
        "trading_mode": "COMBINED" if lgbm_applied else "LOGREG_ONLY",
        "original_strategy": "COMBINED" if lgbm_applied else "LOGREG_ONLY",
        "lightgbm_decision_mode": lgbm_mode,
        "lightgbm_applied": lgbm_applied,
        "decision_source": "LOGREG_PLUS_LIGHTGBM" if lgbm_applied else "LOGREG_ONLY",
        "ml_phase_model": comb_res.entry_model_key,
    }
    lgbm_meta = json.dumps(lgbm_meta_dict)

    if comb_res.action != "SKIP":
        trade_decision = TradeDecision(
            action=comb_res.action,
            buy_price=comb_res.candidate_ask or 0.0,
            bet_size_usdc=comb_res.bet_size_usdc,
            reason=comb_res.reason,
            strategy_type="COMBINED" if lgbm_applied else "LOGREG_ONLY",
            p_flip=comb_res.p_flip,
            p_up=comb_res.direction_probability if lgbm_direction_value == "UP" else (1.0 - comb_res.direction_probability if comb_res.direction_probability is not None else None),
            strike=comb_res.strike_proxy,
            edge=comb_res.net_edge,
            p_win_effective=comb_res.p_candidate_win,
            p_win_raw=comb_res.p_logreg_win if comb_res.p_logreg_win is not None else comb_res.p_candidate_win,
            decision_details=decision_details,
        )
    else:
        trade_decision = TradeDecision(
            action="SKIP",
            buy_price=comb_res.candidate_ask or 0.0,
            bet_size_usdc=0.0,
            reason=comb_res.reason,
            strategy_type="COMBINED" if lgbm_applied else "LOGREG_ONLY",
            p_flip=comb_res.p_flip,
            p_up=comb_res.direction_probability if lgbm_direction_value == "UP" else None,
            strike=comb_res.strike_proxy,
            edge=comb_res.net_edge or 0.0,
            p_win_effective=comb_res.p_candidate_win,
            p_win_raw=comb_res.p_logreg_win if comb_res.p_logreg_win is not None else comb_res.p_candidate_win,
            decision_details=decision_details,
        )

    # 7. Записываем воронку (DecisionFunnelLog) — ровно одна запись!
    is_outsider = (
        (comb_res.candidate_side == "BUY_NO" and fresh_yes_price >= 0.50)
        or (comb_res.candidate_side == "BUY_YES" and fresh_yes_price < 0.50)
    ) if comb_res.candidate_side else False
    min_edge_val = cfg.get_min_edge(is_outsider=is_outsider)

    g1_loaded = bool(entry_model is not None and (lgbm_applied or comb_res.entry_status not in ("MODEL_NOT_FOUND", "MODEL_NOT_LOADED")))
    g2_fetched = True
    if lgbm_applied:
        g3_dir = bool(comb_res.direction_status in ("READY", "DIRECTION_NONE_FALLBACK_LR") and comb_res.entry_status not in ("DIRECTION_UNAVAILABLE", "DIRECTION_VETOED", "LOW_DIRECTION_PROB"))
    else:
        g3_dir = bool(comb_res.entry_status not in ("LOGREG_ABSTAIN", "MODEL_NOT_FOUND") and comb_res.candidate_side in ("BUY_YES", "BUY_NO"))

    g4_consensus = bool(comb_res.candidate_side in ("BUY_YES", "BUY_NO") and comb_res.entry_status != "CONSENSUS_FAILED")
    g5_win_prob = bool(comb_res.p_candidate_win is not None and comb_res.p_candidate_win >= getattr(cfg, "min_win_prob", 0.51) and comb_res.entry_status != "LOW_WIN_PROB")
    g6_price_time = bool(comb_res.entry_status not in ("INVALID_TIME", "OUTSIDER_DISABLED", "FAVORITE_DISABLED", "PRICE_OUT_OF_BOUNDS"))
    g7_net_edge = bool(comb_res.net_edge is not None and comb_res.entry_status != "INSUFFICIENT_NET_EDGE" and comb_res.net_edge >= min_edge_val)
    g7_crypto_confirm = g7_net_edge if lgbm_applied else None
    g8_vote = bool(comb_res.action in ("BUY_YES", "BUY_NO") and comb_res.bet_size_usdc > 0)

    confirm_model_key = applied_direction_key
    confirm_model_version = applied_direction_version
    confirm_passed = (comb_res.direction_status == "READY") if lgbm_applied else None
    final_dir_status = "SHADOW_NOT_APPLIED" if lgbm_shadow else ("DISABLED_BY_OPERATOR" if lgbm_mode == "OFF" else comb_res.direction_status)

    await log_funnel(
        db_session,
        market_id=market.market_id,
        asset=market.asset,
        trading_mode="COMBINED" if lgbm_applied else "LOGREG_ONLY",
        execution_mode=execution_mode,
        decision_run_id=decision_run_id,
        used_model=comb_res.entry_model_key,
        p_flip=comb_res.p_flip,
        edge=comb_res.net_edge,
        fresh_price=comb_res.candidate_ask or fresh_yes_price,
        threshold_lower=1.0 - cfg.flip_threshold,
        threshold_upper=cfg.flip_threshold,
        min_edge_used=min_edge_val,
        g1_model_loaded=g1_loaded,
        g2_price_fetched=g2_fetched,
        g3_dead_zone=g3_dir,
        g4_no_flip=g4_consensus,
        g5_min_edge=g5_win_prob,
        g6_price_range=g6_price_time,
        g7_crypto_confirm=g7_crypto_confirm,
        g8_combined_vote=g8_vote,
        primary_model_key=comb_res.entry_model_key,
        primary_model_version=comb_res.entry_model_version,
        confirm_model_key=confirm_model_key,
        confirm_model_version=confirm_model_version,
        proposed_action=comb_res.action,
        proposed_price=comb_res.candidate_ask,
        proposed_amount_usdc=comb_res.bet_size_usdc if comb_res.action != "SKIP" else 0.0,
        confirm_direction=lgbm_direction_value,
        confirm_passed=confirm_passed,
        
        # Новая телеметрия
        direction_status=final_dir_status,
        direction_model_key=lgbm_attribution["funnel_model_key"],
        direction_model_version=lgbm_attribution["funnel_model_version"],
        direction_regime=comb_res.direction_regime,
        direction_probability=comb_res.direction_probability,
        direction_p_up=comb_res.direction_p_up,
        direction_p_down=comb_res.direction_p_down,
        direction_raw_opinion=comb_res.direction_raw_opinion,
        direction_p_up_raw=comb_res.direction_p_up_raw,
        direction_p_down_raw=comb_res.direction_p_down_raw,
        required_direction_model_key=f"{asset_upper}_{comb_res.direction_regime}" if comb_res.direction_regime else None,
        direction_value=lgbm_direction_value,
        entry_requested_key=phase_asset,
        entry_model_key=comb_res.entry_model_key,
        entry_model_version=comb_res.entry_model_version,
        entry_model_phase=comb_res.entry_model_phase,
        entry_model_source=comb_res.entry_model_source,
        entry_status=comb_res.entry_status,
        fallback_reason=comb_res.fallback_reason,
        p_candidate_win=comb_res.p_candidate_win,
        p_logreg_win=comb_res.p_logreg_win,
        direction_discount_mult=comb_res.direction_discount_applied,
        combined_dir_discount_weight=comb_res.combined_dir_discount_weight,
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
        direction_error_detail=comb_res.direction_error_detail,
        would_live_accept=comb_res.would_live_accept,
        p_flip_raw=comb_res.p_flip_raw,
        entry_model_ece=comb_res.entry_model_ece,

        final_action=comb_res.action,
        skip_reason=comb_res.reason if comb_res.action == "SKIP" else None,
    )

    # ── MRF: apply regime filter to decision ──────────────────────────────
    mrf_adjusted_action = comb_res.action
    mrf_adjusted_bet = comb_res.bet_size_usdc
    mrf_audit = None
    mrf_outcome = None

    if comb_res.action in ("BUY_YES", "BUY_NO") and cfg.mrf_mode != "OFF":
        mrf_adjusted_action, mrf_adjusted_bet, mrf_audit, mrf_outcome = await _apply_mrf_filter(
            db_session=db_session,
            cfg=cfg,
            asset_upper=asset_upper,
            binance_symbol=binance_symbol or "",
            start_time=start_time,
            candidate_side=comb_res.candidate_side,
            fresh_yes_price=fresh_yes_price,
            bet_size_usdc=comb_res.bet_size_usdc,
            action=comb_res.action,
            decision_run_id=decision_run_id,
            lgbm_applied=bool(entry_model_phase and entry_model_phase != "FAILED"),
        )

    # Update trade_decision if MRF changed action OR bet size
    if mrf_adjusted_action != comb_res.action or mrf_adjusted_bet != comb_res.bet_size_usdc:
        mrf_phase = "UNKNOWN"
        if mrf_outcome and mrf_outcome.global_phase:
            mrf_phase = mrf_outcome.global_phase
        elif mrf_audit and isinstance(mrf_audit, dict):
            mrf_phase = mrf_audit.get("global_phase", "UNKNOWN")
        skip_reason = mrf_outcome.skip_reason if mrf_outcome and mrf_outcome.skip_reason else f"MRF:{mrf_phase}"
        logger.info(
            "mrf_decision_override",
            asset=asset_upper,
            original=comb_res.action,
            adjusted=mrf_adjusted_action,
            original_bet=comb_res.bet_size_usdc,
            adjusted_bet=mrf_adjusted_bet,
            phase=mrf_phase,
        )
        trade_decision = TradeDecision(
            action=mrf_adjusted_action,
            buy_price=trade_decision.buy_price,
            bet_size_usdc=mrf_adjusted_bet,
            reason=skip_reason if mrf_adjusted_action == "SKIP" else trade_decision.reason,
            strategy_type=trade_decision.strategy_type,
            p_flip=trade_decision.p_flip,
            p_up=trade_decision.p_up,
            strike=trade_decision.strike,
            edge=trade_decision.edge,
            p_win_effective=trade_decision.p_win_effective,
            p_win_raw=trade_decision.p_win_raw,
            decision_details=trade_decision.decision_details,
        )

    # Build MRF audit for caller
    mrf_audit_dict = None
    if mrf_outcome and mrf_outcome.audit_dict:
        mrf_audit_dict = mrf_outcome.audit_dict

    return DecisionResult(
        decision_obj=trade_decision,
        p_flip=comb_res.p_flip if comb_res.p_flip is not None else 0.0,
        model_ver=comb_res.entry_model_version,
        edge=comb_res.net_edge,
        skip_reason=comb_res.reason if comb_res.action == "SKIP" else None,
        lgbm_metadata=lgbm_meta,
        used_model_key=comb_res.entry_model_key,
        confirm_model_key=confirm_model_key,
        confirm_model_version=confirm_model_version,
    )
