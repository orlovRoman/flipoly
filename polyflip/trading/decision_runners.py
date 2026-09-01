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
from polyflip.trading.weighted_telemetry import weighted_telemetry_from_object
from polyflip.crypto.market_regime_integration import build_snapshot_from_candles
from polyflip.crypto.market_regime import MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_apply import apply_market_regime_filter
from polyflip.crypto.market_regime_classifier import Regime

logger = structlog.get_logger(__name__)


def _resolve_entry_flip_threshold(
    models_cache: Any,
    model_key: Optional[str],
    fallback: float,
) -> float:
    """Return an explicitly opted-in model threshold, otherwise the global fallback."""
    raw_threshold: Any = None
    thresholds = getattr(models_cache, "thresholds", None)
    if model_key and isinstance(thresholds, dict):
        raw_threshold = thresholds.get(model_key)
    if raw_threshold is None:
        raw_threshold = fallback
    try:
        value = float(raw_threshold)
        if value > 1.0:
            value /= 100.0
        if 0.0 <= value <= 1.0:
            return value
    except (TypeError, ValueError, OverflowError):
        pass
    return float(fallback)


def _resolve_lgbm_attribution(
    mode: str,
    direction_value: Any,
    model_key: Optional[str],
    model_version: Optional[int],
    probability_applied: bool = False,
) -> dict[str, Any]:
    """Normalize LGBM direction telemetry and separate applied vs observed attribution.

    A loaded model is not an applied directional model when it abstains (NONE or an
    empty value).  The weighted policy is an exception: it can apply the model's
    calibrated probability without using its legacy UP/DOWN threshold.  Shadow mode
    still exposes the loaded model for telemetry, while ``direction_model_key`` in
    active/funnel attribution identifies a model used by the active policy.
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
    actually_decided = normalized_value in {"UP", "DOWN"} or bool(probability_applied)
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



def _build_mrf_failure_audit(
    cfg,
    failure_reason: str,
    reason_codes: list[str] | None = None,
    as_of=None,
) -> dict:
    """Build a serializable audit for attempts that could not classify MRF."""
    return {
        "mode": getattr(cfg, "mrf_mode", "UNKNOWN"),
        "version": getattr(cfg, "mrf_version", 2),
        "global_phase": "UNKNOWN",
        "global_regime": "UNKNOWN",
        "asset_phase": "UNKNOWN",
        "asset_regime": "UNKNOWN",
        "global_strength": 0.0,
        "global_confidence": 0.0,
        "applied": False,
        "history_ready": False,
        "failure_reason": failure_reason,
        "reason_codes": list(reason_codes or []),
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of),
    }


async def _load_mrf_snapshot(
    db_session,
    cfg,
    start_time,
    asset_upper: str = "",
) -> tuple[Any | None, dict | None, str | None]:
    """Load and validate the shared multi-asset MRF snapshot once.

    Weighted policy may need the regime evidence before it chooses a side,
    while the legacy MRF gate is applied after the decision.  Returning the
    snapshot and the failure audit lets both stages use the same candle
    window without issuing a second set of database reads.
    """
    try:
        from polyflip.constants import COMBINED_BINANCE_SYMBOLS, COMBINED_MODE_SUPPORTED_ASSETS
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles

        limit = max(cfg.mrf_min_history, MIN_HISTORY_CANDLES) + 10
        tasks = {}
        for asset_name in COMBINED_MODE_SUPPORTED_ASSETS:
            sym = COMBINED_BINANCE_SYMBOLS.get(asset_name)
            if sym:
                tasks[asset_name] = get_recent_candles(
                    db_session, sym, "15m", limit=limit,
                )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        candles_by_asset = {}
        fetch_errors = []
        for asset_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning("mrf_candle_fetch_error", asset=asset_name, error=str(result))
                fetch_errors.append(asset_name)
                continue
            if result:
                candles_by_asset[asset_name] = result

        if not candles_by_asset:
            logger.warning("mrf_no_candles_any_asset")
            failure = "candle_error:no_candles"
            return None, _build_mrf_failure_audit(cfg, failure, as_of=start_time), failure

        snapshot = build_snapshot_from_multi_asset_candles(
            candles_by_asset,
            as_of=start_time,
            expected_assets=list(COMBINED_MODE_SUPPORTED_ASSETS),
        )

        if not snapshot.basket.history_ready:
            reason_codes = snapshot.reason_codes or []
            if any("asset_missing" in r for r in reason_codes):
                failure = "missing_asset:" + ",".join(
                    r.split(":", 1)[1] for r in reason_codes if "asset_missing" in r
                )
            elif any("no_candles" in r for r in reason_codes):
                failure = "candle_error:" + ",".join(
                    r.split(":", 1)[1] for r in reason_codes if "no_candles" in r
                )
            elif any("candle_continuity" in r for r in reason_codes):
                failure = "continuity_error:" + ",".join(
                    r.split(":", 1)[1] for r in reason_codes if "candle_continuity" in r
                )
            elif any("insufficient_history" in r for r in reason_codes):
                failure = "insufficient_history"
            else:
                failure = "not_ready"
            logger.info(
                "mrf_history_not_ready",
                asset=asset_upper,
                reason_codes=reason_codes,
                failure_reason=failure,
            )
            return (
                None,
                _build_mrf_failure_audit(
                    cfg, failure, reason_codes=reason_codes, as_of=start_time,
                ),
                failure,
            )

        return snapshot, None, None
    except Exception as exc:
        logger.error("mrf_snapshot_load_error", asset=asset_upper, error=str(exc))
        failure = f"runtime_error:{type(exc).__name__}"
        return None, _build_mrf_failure_audit(cfg, failure, as_of=start_time), failure


def _weighted_mrf_yes_evidence(snapshot: Any, cfg, asset_upper: str) -> float:
    """Return MRF evidence oriented to the YES/UP side for the scorer.

    ``RegimeGateResult.regime_evidence`` is signed relative to a candidate
    direction.  Evaluating it with ``+1`` makes the result a stable YES-axis
    signal; the weighted scorer then naturally applies the opposite sign to a
    BUY_NO candidate.
    """
    from polyflip.crypto.market_regime_apply import _build_regime_config
    from polyflip.crypto.market_regime_policy import VetoGateConfig, evaluate_veto_gate

    gate = evaluate_veto_gate(
        snapshot=snapshot,
        asset_symbol=asset_upper,
        candidate_direction=1.0,
        net_edge=0.0,
        min_edge_used=0.0,
        config=VetoGateConfig(
            asset_weight=cfg.mrf_asset_weight,
            global_weight=cfg.mrf_global_weight,
            veto_threshold=cfg.mrf_veto_threshold,
            edge_override_margin=cfg.mrf_edge_override_margin,
        ),
        regime_config=_build_regime_config(cfg),
    )
    return float(gate.regime_evidence)


async def _apply_mrf_filter(
    db_session,
    cfg,
    asset_upper: str,
    binance_symbol: str,
    start_time,
    candidate_side: str | None,
    fresh_yes_price: float,
    candidate_ask: float,
    bet_size_usdc: float,
    net_edge: float,
    min_edge_used: float,
    action: str,
    decision_run_id: str = "",
    lgbm_applied: bool = False,
    preloaded_snapshot: Any | None = None,
    preloaded_audit: dict | None = None,
    preloaded_failure_reason: str | None = None,
):
    """Apply MRF filter. Returns (adjusted_action, adjusted_bet_size, mrf_audit, outcome, failure_reason)."""
    if cfg.mrf_mode == "OFF":
        return action, bet_size_usdc, None, None, None

    # A v3 gate only has a meaningful direction for a concrete BUY side.
    # Do not let an inconsistent decision silently bypass an ACTIVE veto.
    if cfg.mrf_version == 3 and candidate_side not in {"BUY_YES", "BUY_NO"}:
        side_label = candidate_side or "NONE"
        failure = f"invalid_candidate_side:{side_label}"
        logger.error("mrf_invalid_candidate_side", asset=asset_upper, candidate_side=side_label)
        audit = _build_mrf_failure_audit(cfg, failure, as_of=start_time)
        if cfg.mrf_mode == "ACTIVE":
            return "SKIP", 0.0, audit, None, failure
        return action, bet_size_usdc, audit, None, failure

    try:
        if preloaded_snapshot is not None:
            snapshot = preloaded_snapshot
            preloaded_failure_reason = None
        elif preloaded_failure_reason:
            failure = preloaded_failure_reason
            return (
                action,
                bet_size_usdc,
                preloaded_audit or _build_mrf_failure_audit(cfg, failure, as_of=start_time),
                None,
                failure,
            )
        else:
            snapshot, loaded_audit, loaded_failure = await _load_mrf_snapshot(
                db_session, cfg, start_time, asset_upper,
            )
            if snapshot is None:
                failure = loaded_failure or "not_ready"
                return (
                    action,
                    bet_size_usdc,
                    loaded_audit or _build_mrf_failure_audit(cfg, failure, as_of=start_time),
                    None,
                    failure,
                )

        outcome = apply_market_regime_filter(
            cfg=cfg,
            snapshot=snapshot,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            fresh_yes_price=fresh_yes_price,
            lgbm_applied=lgbm_applied,
            bet_size_usdc=bet_size_usdc,
            net_edge=net_edge,
            min_edge_used=min_edge_used,
            action=action,
            decision_run_id=decision_run_id,
            asset_symbol=asset_upper,
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

        return outcome.adjusted_action, outcome.adjusted_bet_size, outcome.audit_dict, outcome, None

    except Exception as exc:
        logger.error("mrf_error", asset=asset_upper, error=str(exc))
        failure = f"runtime_error:{type(exc).__name__}"
        return (
            action,
            bet_size_usdc,
            _build_mrf_failure_audit(cfg, failure, as_of=start_time),
            None,
            failure,
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
    no_best_ask = round(1.0 - yes_best_ask, 8)  # mirror of YES best ask as NO fallback
    if market.no_token_id:
        fresh_no_prices = await api_client.get_market_prices(market.no_token_id)
        if fresh_no_prices and fresh_no_prices.get("best_ask") is not None:
            no_best_ask = float(fresh_no_prices["best_ask"])
            if fresh_no_prices.get("current_yes_price") is not None:
                fresh_no_price = float(fresh_no_prices["current_yes_price"])

    # Weighted policy may use the market's feeSchedule.  If the metadata is
    # unavailable, keep the explicit configured fallback and record its
    # provenance in decision_details.
    weighted_fee_rate = None
    weighted_fee_exponent = None
    weighted_fee_source = "CONFIG_DEFAULT"
    policy_mode = str(getattr(cfg, "trading_policy_mode", "LEGACY") or "LEGACY").upper()
    condition_id = getattr(market, "condition_id", None)
    if (
        policy_mode in {"WEIGHTED_SHADOW", "WEIGHTED_ACTIVE"}
        and isinstance(condition_id, (str, int))
        and str(condition_id).strip()
        and hasattr(api_client, "get_market_fee_schedule")
    ):
        try:
            fee_schedule = await api_client.get_market_fee_schedule(str(condition_id))
            if isinstance(fee_schedule, dict):
                if fee_schedule.get("fees_enabled") is False:
                    weighted_fee_rate = 0.0
                    weighted_fee_source = "CLOB_FEE_SCHEDULE_DISABLED"
                elif fee_schedule.get("fee_rate") is not None:
                    weighted_fee_rate = float(fee_schedule["fee_rate"])
                    weighted_fee_source = str(
                        fee_schedule.get("source") or "CLOB_FEE_SCHEDULE"
                    )
                if fee_schedule.get("fee_exponent") is not None:
                    weighted_fee_exponent = float(fee_schedule["fee_exponent"])
        except (TypeError, ValueError, OverflowError) as exc:
            logger.debug(
                "weighted_fee_schedule_parse_failed",
                asset=asset_upper,
                condition_id=str(condition_id),
                error=str(exc),
            )
        except Exception as exc:
            logger.debug(
                "weighted_fee_schedule_lookup_failed",
                asset=asset_upper,
                condition_id=str(condition_id),
                error=str(exc),
            )

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
    entry_flip_threshold: Optional[float] = None

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
        entry_flip_threshold = _resolve_entry_flip_threshold(
            models_cache, candidate_key, cfg.flip_threshold
        )
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

    # Preload regime evidence for weighted telemetry even when beta is zero.
    # WEIGHTED_ACTIVE consumes it as a soft score input, never as the legacy
    # stake multiplier/veto below.
    weighted_mrf_evidence = None
    preloaded_mrf_snapshot = None
    preloaded_mrf_audit = None
    preloaded_mrf_failure_reason = None
    if (
        policy_mode in {"WEIGHTED_SHADOW", "WEIGHTED_ACTIVE"}
        and cfg.mrf_mode != "OFF"
    ):
        (
            preloaded_mrf_snapshot,
            preloaded_mrf_audit,
            preloaded_mrf_failure_reason,
        ) = await _load_mrf_snapshot(
            db_session, cfg, start_time, asset_upper,
        )
        if preloaded_mrf_snapshot is not None:
            try:
                weighted_mrf_evidence = _weighted_mrf_yes_evidence(
                    preloaded_mrf_snapshot, cfg, asset_upper,
                )
            except Exception as exc:
                logger.debug(
                    "weighted_mrf_evidence_unavailable",
                    asset=asset_upper,
                    error=str(exc),
                )

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
        flip_threshold=entry_flip_threshold,
        mrf_evidence=weighted_mrf_evidence,
        spread=0.0,  # yes_best_ask/no_best_ask are executable asks, not midpoints
        weighted_fee_rate=weighted_fee_rate,
        weighted_fee_exponent=weighted_fee_exponent,
        weighted_fee_source=weighted_fee_source,
    )

    elapsed = time.monotonic() - t0
    logger.info("combined_mode_latency", asset=asset_upper, elapsed_ms=round(elapsed * 1000, 1))

    lgbm_mode = getattr(cfg, "lightgbm_decision_mode", "SHADOW")
    # In weighted ACTIVE, a valid LightGBM probability is an applied input
    # even when the legacy LightGBM switch remains SHADOW.  Without this
    # distinction telemetry and MRF strategy attribution would incorrectly
    # label a weighted decision as LogReg-only.
    weighted_lgbm_used = (
        policy_mode == "WEIGHTED_ACTIVE"
        and comb_res.weighted_p_lgbm_yes is not None
    )
    effective_lgbm_mode = "ACTIVE" if weighted_lgbm_used else lgbm_mode
    lgbm_applied = (effective_lgbm_mode == "ACTIVE")
    lgbm_shadow = (effective_lgbm_mode == "SHADOW")
    lgbm_attribution = _resolve_lgbm_attribution(
        effective_lgbm_mode,
        comb_res.direction_value,
        comb_res.direction_model_key,
        comb_res.direction_model_version,
        probability_applied=weighted_lgbm_used,
    )
    lgbm_direction_value = lgbm_attribution["direction_value"]
    applied_direction_key = lgbm_attribution["applied_model_key"]
    applied_direction_version = lgbm_attribution["applied_model_version"]

    # 6. Формируем decision_details и TradeDecision
    decision_details = {
        "decision_run_id": decision_run_id,
        "lightgbm_decision_mode": lgbm_mode,
        "lightgbm_applied": lgbm_applied,
        "decision_source": (
            "WEIGHTED_POLICY"
            if getattr(cfg, "trading_policy_mode", "LEGACY") == "WEIGHTED_ACTIVE"
            else ("LOGREG_PLUS_LIGHTGBM" if lgbm_applied else "LOGREG_ONLY")
        ),
        "weighted_policy_mode": comb_res.weighted_policy_mode,
        "weighted_policy_id": comb_res.weighted_policy_id,
        "weighted_p_market_yes": comb_res.weighted_p_market_yes,
        "weighted_p_logreg_yes": comb_res.weighted_p_logreg_yes,
        "weighted_p_lgbm_yes": comb_res.weighted_p_lgbm_yes,
        "weighted_p_final_yes": comb_res.weighted_p_final_yes,
        "weighted_market_weight": comb_res.weighted_market_weight,
        "weighted_logreg_weight": comb_res.weighted_logreg_weight,
        "weighted_lgbm_weight": comb_res.weighted_lgbm_weight,
        "weighted_mrf_evidence": comb_res.weighted_mrf_evidence,
        "weighted_market_contribution_logodds": comb_res.weighted_market_contribution_logodds,
        "weighted_logreg_contribution_logodds": comb_res.weighted_logreg_contribution_logodds,
        "weighted_lgbm_contribution_logodds": comb_res.weighted_lgbm_contribution_logodds,
        "weighted_mrf_contribution_logodds": comb_res.weighted_mrf_contribution_logodds,
        "weighted_intercept_contribution_logodds": comb_res.weighted_intercept_contribution_logodds,
        "weighted_models_agree": comb_res.weighted_models_agree,
        "weighted_selected_side": comb_res.weighted_selected_side,
        "weighted_yes_net_ev": comb_res.weighted_yes_net_ev,
        "weighted_no_net_ev": comb_res.weighted_no_net_ev,
        "weighted_net_ev_per_share": comb_res.weighted_net_ev_per_share,
        "weighted_cost_per_share": comb_res.weighted_cost_per_share,
        "weighted_fee_rate": comb_res.weighted_fee_rate,
        "weighted_fee_exponent": comb_res.weighted_fee_exponent,
        "weighted_fee_per_share": comb_res.weighted_fee_per_share,
        "weighted_slippage_per_share": comb_res.weighted_slippage_per_share,
        "weighted_missing_components": comb_res.weighted_missing_components,
        "weighted_selection_reason": comb_res.weighted_selection_reason,
        "weighted_fee_source": comb_res.weighted_fee_source,
        "consensus_type": comb_res.consensus_type,
        "direction_status": "SHADOW_NOT_APPLIED" if lgbm_shadow else ("DISABLED_BY_OPERATOR" if effective_lgbm_mode == "OFF" else comb_res.direction_status),
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
        "entry_flip_threshold": entry_flip_threshold,
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
        "decision_source": (
            "WEIGHTED_POLICY"
            if comb_res.weighted_policy_mode == "WEIGHTED_ACTIVE"
            else ("LOGREG_PLUS_LIGHTGBM" if lgbm_applied else "LOGREG_ONLY")
        ),
        "weighted_policy_mode": comb_res.weighted_policy_mode,
        "weighted_policy_id": comb_res.weighted_policy_id,
        "weighted_p_final_yes": comb_res.weighted_p_final_yes,
        "weighted_selected_side": comb_res.weighted_selected_side,
        "weighted_net_ev_per_share": comb_res.weighted_net_ev_per_share,
        "weighted_cost_per_share": comb_res.weighted_cost_per_share,
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
            direction_value=comb_res.direction_value,
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
            direction_value=comb_res.direction_value,
        )

    # 7. Записываем воронку (DecisionFunnelLog) — ровно одна запись!
    is_outsider = (
        (comb_res.candidate_side == "BUY_NO" and fresh_yes_price >= 0.50)
        or (comb_res.candidate_side == "BUY_YES" and fresh_yes_price < 0.50)
    ) if comb_res.candidate_side else False
    min_edge_val = (
        cfg.get_weighted_min_net_ev(is_outsider)
        if policy_mode == "WEIGHTED_ACTIVE"
        else cfg.get_min_edge(is_outsider=is_outsider)
    )

    g1_loaded = bool(entry_model is not None and (lgbm_applied or comb_res.entry_status not in ("MODEL_NOT_FOUND", "MODEL_NOT_LOADED")))
    g2_fetched = True
    if weighted_lgbm_used:
        g3_dir = bool(
            comb_res.weighted_p_lgbm_yes is not None
            and comb_res.entry_status not in ("DIRECTION_UNAVAILABLE", "DIRECTION_VETOED")
        )
    elif lgbm_applied:
        g3_dir = bool(comb_res.direction_status in ("READY", "DIRECTION_NONE_FALLBACK_LR") and comb_res.entry_status not in ("DIRECTION_UNAVAILABLE", "DIRECTION_VETOED", "LOW_DIRECTION_PROB"))
    else:
        g3_dir = bool(comb_res.entry_status not in ("LOGREG_ABSTAIN", "MODEL_NOT_FOUND") and comb_res.candidate_side in ("BUY_YES", "BUY_NO"))

    g4_consensus = bool(comb_res.candidate_side in ("BUY_YES", "BUY_NO") and comb_res.entry_status != "CONSENSUS_FAILED")
    g5_win_prob = bool(
        comb_res.p_candidate_win is not None
        and (policy_mode == "WEIGHTED_ACTIVE" or comb_res.p_candidate_win >= getattr(cfg, "min_win_prob", 0.51))
        and comb_res.entry_status != "LOW_WIN_PROB"
    )
    g6_price_time = bool(comb_res.entry_status not in ("INVALID_TIME", "OUTSIDER_DISABLED", "FAVORITE_DISABLED", "PRICE_OUT_OF_BOUNDS"))
    g7_net_edge = bool(comb_res.net_edge is not None and comb_res.entry_status != "INSUFFICIENT_NET_EDGE" and comb_res.net_edge >= min_edge_val)
    g7_crypto_confirm = g7_net_edge if lgbm_applied else None
    g8_vote = bool(comb_res.action in ("BUY_YES", "BUY_NO") and comb_res.bet_size_usdc > 0)

    confirm_model_key = applied_direction_key
    confirm_model_version = applied_direction_version
    confirm_passed = (
        True
        if weighted_lgbm_used and comb_res.weighted_p_lgbm_yes is not None
        else ((comb_res.direction_status == "READY") if lgbm_applied else None)
    )
    final_dir_status = "SHADOW_NOT_APPLIED" if lgbm_shadow else ("DISABLED_BY_OPERATOR" if lgbm_mode == "OFF" else comb_res.direction_status)

    # ── MRF: apply regime filter BEFORE funnel logging (MRF-FIX-08) ──────
    original_action = comb_res.action
    original_bet = comb_res.bet_size_usdc
    mrf_adjusted_action = comb_res.action
    mrf_adjusted_bet = comb_res.bet_size_usdc
    mrf_audit = None
    mrf_outcome = None
    mrf_failure_reason = None
    mrf_pre_outcome_reason = None

    if (
        comb_res.action in ("BUY_YES", "BUY_NO")
        and cfg.mrf_mode != "OFF"
        and policy_mode != "WEIGHTED_ACTIVE"
    ):
        mrf_adjusted_action, mrf_adjusted_bet, mrf_audit, mrf_outcome, mrf_pre_outcome_reason = await _apply_mrf_filter(
            db_session=db_session,
            cfg=cfg,
            asset_upper=asset_upper,
            binance_symbol=binance_symbol or "",
            start_time=start_time,
            candidate_side=comb_res.candidate_side,
            fresh_yes_price=fresh_yes_price,
            candidate_ask=comb_res.candidate_ask or fresh_yes_price,
            bet_size_usdc=comb_res.bet_size_usdc,
            net_edge=comb_res.net_edge or 0.0,
            min_edge_used=min_edge_val,
            action=comb_res.action,
            decision_run_id=decision_run_id,
            lgbm_applied=lgbm_applied,  # MRF-FIX-07: real lgbm flag from lgbm_mode == "ACTIVE"
            preloaded_snapshot=preloaded_mrf_snapshot,
            preloaded_audit=preloaded_mrf_audit,
            preloaded_failure_reason=preloaded_mrf_failure_reason,
        )
        if mrf_outcome and mrf_outcome.skip_reason:
            mrf_failure_reason = mrf_outcome.skip_reason
        elif mrf_pre_outcome_reason:
            # DecisionFunnelLog.mrf_failure_reason is VARCHAR(256).
            mrf_failure_reason = str(mrf_pre_outcome_reason)[:256]

    # Update trade_decision if MRF changed action OR bet size
    mrf_phase = "UNKNOWN"
    mrf_asset_phase_str = "UNKNOWN"
    mrf_strength_val = 0.0
    mrf_confidence_val = 0.0
    mrf_multiplier_val = None
    mrf_policy_version_val = None
    mrf_regime_evidence_val = None
    mrf_gate_threshold_val = None
    mrf_edge_margin_val = None
    mrf_gate_would_block_val = None
    mrf_gate_reason_val = None

    if mrf_outcome:
        mrf_phase = mrf_outcome.global_phase or "UNKNOWN"
        mrf_asset_phase_str = mrf_outcome.asset_phase or "UNKNOWN"
        mrf_policy_version_val = mrf_outcome.policy_version
        if mrf_outcome.policy_result:
            mrf_strength_val = mrf_outcome.policy_result.global_strength
            mrf_confidence_val = mrf_outcome.policy_result.global_confidence
            mrf_multiplier_val = mrf_outcome.policy_result.stake_multiplier
        if mrf_outcome.gate_result is not None:
            gate = mrf_outcome.gate_result
            mrf_strength_val = gate.global_strength
            mrf_confidence_val = gate.global_confidence
            mrf_regime_evidence_val = gate.regime_evidence
            mrf_gate_threshold_val = gate.veto_threshold
            mrf_edge_margin_val = gate.edge_margin
            mrf_gate_would_block_val = gate.would_block
            mrf_gate_reason_val = gate.reason
            # v3 is binary and must never be represented as a legacy
            # stake multiplier.
            mrf_multiplier_val = None
    elif mrf_audit and isinstance(mrf_audit, dict):
        mrf_phase = mrf_audit.get("global_phase", "UNKNOWN")
        mrf_policy_version_val = mrf_audit.get("version")

    if mrf_adjusted_action != comb_res.action or mrf_adjusted_bet != comb_res.bet_size_usdc:
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
            probability_adjustment=trade_decision.probability_adjustment,
            decision_details=trade_decision.decision_details,
            direction_value=trade_decision.direction_value,
        )

    # Build MRF audit JSON for funnel
    mrf_audit_json = None
    if mrf_audit and isinstance(mrf_audit, dict):

        mrf_audit_json = json.dumps(mrf_audit, ensure_ascii=False, default=str)

    # ── Log funnel (MRF-FIX-08: now includes MRF results) ─────────────────
    # Step 5: mrf_evaluated=true ONLY if MRF actually classified and evaluated.
    # If _apply_mrf_filter returned None outcome (candle error, not_ready, etc),
    # mrf_evaluated should be false — it was attempted but not truly evaluated.
    mrf_actually_evaluated = (
        mrf_outcome is not None
        and cfg.mrf_mode != "OFF"
        and comb_res.action in ("BUY_YES", "BUY_NO")
    )
    mrf_evaluated = mrf_actually_evaluated

    funnel_flip_threshold = _resolve_entry_flip_threshold(
        models_cache, comb_res.entry_model_key, cfg.flip_threshold
    )

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
        threshold_lower=1.0 - funnel_flip_threshold,
        threshold_upper=funnel_flip_threshold,
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
        **weighted_telemetry_from_object(comb_res),

        # MRF telemetry (MRF-FIX-03 + MRF-FIX-08)
        mrf_mode=cfg.mrf_mode,
        mrf_phase=mrf_phase,
        mrf_asset_phase=mrf_asset_phase_str,
        mrf_strength=mrf_strength_val,
        mrf_confidence=mrf_confidence_val,
        mrf_multiplier=mrf_multiplier_val,
        mrf_applied=(
            # v3 is a binary veto.  ``RegimeDecisionOutcome.applied`` is
            # intentionally mode-aware (False in SHADOW), while this field
            # records whether the final decision was actually changed.  Use
            # the action diff so a future v3 implementation cannot report a
            # veto as applied merely because an internal flag was set.
            mrf_adjusted_action != original_action
            if mrf_outcome is not None and mrf_outcome.policy_version == 3
            else mrf_adjusted_action != original_action
            or mrf_adjusted_bet != original_bet
        ),
        mrf_evaluated=mrf_evaluated,
        mrf_as_of=start_time,
        mrf_failure_reason=mrf_failure_reason,
        mrf_audit_json=mrf_audit_json,
        mrf_original_action=original_action,
        mrf_original_bet=original_bet,
        mrf_final_action=mrf_adjusted_action,
        mrf_final_bet=mrf_adjusted_bet,
        mrf_policy_version=mrf_policy_version_val,
        mrf_regime_evidence=mrf_regime_evidence_val,
        mrf_gate_threshold=mrf_gate_threshold_val,
        mrf_edge_margin=mrf_edge_margin_val,
        mrf_gate_would_block=mrf_gate_would_block_val,
        mrf_gate_reason=mrf_gate_reason_val,

        final_action=mrf_adjusted_action,
        skip_reason=trade_decision.reason if trade_decision.action == "SKIP" else None,
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
        # ``comb_res`` is the pre-MRF decision.  When v3 actively vetoes a
        # BUY, the final ``trade_decision`` is the source of truth for the
        # returned skip reason.
        skip_reason=trade_decision.reason if trade_decision.action == "SKIP" else None,
        lgbm_metadata=lgbm_meta,
        used_model_key=comb_res.entry_model_key,
        confirm_model_key=confirm_model_key,
        confirm_model_version=confirm_model_version,
    )
