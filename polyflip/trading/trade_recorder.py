import dataclasses
from datetime import datetime
from typing import Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.db.models import LiveMarket, TradeHistory, SlippageLog
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.pre_trade_validator import PreTradeValidation
from polyflip.trading.stoploss import compute_stop_price
from polyflip.trading.takeprofit import compute_take_profit_price
from polyflip.api.trading_dashboard import invalidate_stats_cache
from polyflip.api.dashboard import invalidate_dashboard_cache
from polyflip.execution.outbox import enqueue_open_request, EnqueueDisposition
from polyflip.execution.config import ExecutionSettings
import os
from polyflip.constants import TRADING_MODE_LIGHTGBM, TRADING_MODE_ML, TRADING_MODE_FAVORITE, TRADING_MODE_COMBINED

logger = structlog.get_logger(__name__)

class EnqueueRejected(Exception):
    pass

def _get_trade_active_features(asset_mode: str, active_features_str: str, decision_obj: Any, asset_name: str = "") -> str:
    if asset_mode == TRADING_MODE_LIGHTGBM:
        return "LIGHTGBM_TREND"
    if asset_mode == TRADING_MODE_COMBINED:
        from polyflip.constants import COMBINED_MODE_SUPPORTED_ASSETS
        if not asset_name or asset_name.upper() in COMBINED_MODE_SUPPORTED_ASSETS:
            base = "COMBINED_ML_LGBM"
            if decision_obj and hasattr(decision_obj, "strategy_type") and decision_obj.strategy_type:
                return f"{base},{decision_obj.strategy_type.lower()}"
            return base
        # Если актив не поддерживается, значит был fallback на ML.
        # Fall through к ML-ветке ниже.
        pass
    
    if asset_mode == TRADING_MODE_FAVORITE:
        return "PURE_FAVORITE"
    
    base = active_features_str.strip().rstrip(',') if active_features_str else ""
    if decision_obj and hasattr(decision_obj, "strategy_type") and decision_obj.strategy_type:
        strat = decision_obj.strategy_type.lower()
        return f"{base},{strat}" if base else strat
    return base

async def save_or_update_skipped_trade(
    db_session: AsyncSession,
    market,
    reason: str,
    p_flip_val: float,
    model_version: Optional[int],
    start_time: datetime,
    existing_skipped: Optional[TradeHistory] = None,
    edge: Optional[float] = None,
    active_features: str = "",
    lgbm_metadata: Optional[str] = None,
    market_role: Optional[str] = None,
    *,
    model_key: Optional[str] = None,
    confirm_model_key: Optional[str] = None,
    confirm_model_version: Optional[int] = None,
):
    """Сохраняет запись о пропуске сделки в БД или обновляет её причину."""
    if existing_skipped:
        if (existing_skipped.error_msg != reason or 
            existing_skipped.predicted_flip_prob != p_flip_val or 
            existing_skipped.edge != edge or
            existing_skipped.active_features != active_features or
            existing_skipped.lgbm_metadata != lgbm_metadata or
            (market_role and existing_skipped.market_role != market_role) or
            (model_key and existing_skipped.model_key != model_key)):
            existing_skipped.error_msg = reason
            existing_skipped.predicted_flip_prob = p_flip_val
            existing_skipped.model_version = model_version
            existing_skipped.edge = edge
            if active_features:
                existing_skipped.active_features = active_features
            existing_skipped.lgbm_metadata = lgbm_metadata
            if market_role:
                existing_skipped.market_role = market_role
            if model_key:
                existing_skipped.model_key = model_key
                existing_skipped.model_attribution_source = "EXACT"
            if confirm_model_key:
                existing_skipped.confirm_model_key = confirm_model_key
                existing_skipped.confirm_model_version = confirm_model_version
            existing_skipped.updated_at = start_time
    else:
        history = TradeHistory(
            market_id=market.market_id,
            asset=market.asset,
            outcome_bought="NONE",
            amount_usdc=0.0,
            executed_price=0.0,
            predicted_flip_prob=p_flip_val,
            active_features=active_features,
            model_version=model_version,
            status="SKIPPED",
            error_msg=reason,
            market_role=market_role or "FAVORITE",
            mode="LIVE" if bool(os.getenv("POLYGON_PRIVATE_KEY") and os.getenv("POLYGON_ADDRESS")) else "PAPER",
            edge=edge,
            lgbm_metadata=lgbm_metadata,
            model_key=model_key,
            confirm_model_key=confirm_model_key,
            confirm_model_version=confirm_model_version,
            model_attribution_source="EXACT" if model_key else None,
            created_at=start_time
        )
        db_session.add(history)



async def execute_and_record(
    db_session: AsyncSession,
    market: LiveMarket,
    decision_obj: TradeDecision,
    validation: PreTradeValidation,
    asset_mode: str,
    active_features: str,
    p_flip: float,
    model_ver: Optional[int],
    cfg: TradingConfig,
    existing_skipped: Optional[TradeHistory],
    start_time: datetime,
    lgbm_metadata: Optional[str] = None,
    *,
    model_key: Optional[str] = None,
    confirm_model_key: Optional[str] = None,
    confirm_model_version: Optional[int] = None,
) -> None:
    if not validation.valid:
        raise ValueError("execute_and_record called with failed validation")
        
    decision = decision_obj.action.replace("BUY_", "")
    buy_price = validation.buy_price
    actual_bet_size = validation.actual_bet_size
    edge = validation.edge
    
    logger.info(
        "trade_decision",
        asset=market.asset,
        market_id=market.market_id,
        action=decision_obj.action,
        p_flip=round(p_flip, 4) if p_flip is not None else None,
        p_up=round(decision_obj.p_up, 4) if decision_obj.p_up is not None else None,
        strike=decision_obj.strike,
        edge=round(edge, 4) if edge is not None else None,
        buy_price=buy_price,
        strategy=decision_obj.strategy_type,
        bet_size=actual_bet_size
    )

    if existing_skipped:
        await db_session.delete(existing_skipped)

    # Снимаем паспорт настроек на момент совершения сделки
    try:
        import json
        from polyflip.services.preset_service import PresetService
        config_snap = await PresetService.capture_snapshot(db_session)
        config_snap["_trade_context"] = {
            "time_left_min": getattr(market, "time_left_min", None),
            "current_spread": getattr(market, "spread", None),
            "edge_at_entry": round(edge, 4) if edge is not None else None,
            "p_flip_at_entry": round(p_flip, 4) if p_flip is not None else None,
            "buy_price": buy_price,
            "recorded_at_utc": start_time.isoformat(),
            "decision_details": getattr(decision_obj, "decision_details", None)
        }
        config_snapshot_json = json.dumps(config_snap, ensure_ascii=False)
    except Exception as exc_snap:
        logger.warning("trade_config_snapshot_failed", error=str(exc_snap))
        config_snapshot_json = None

    market_role = validation.market_role
    if market_role not in {"FAVORITE", "OUTSIDER"}:
        raise ValueError("Pre-trade market role is missing")
    p_flip_effective = decision_obj.decision_details.get("p_flip_effective") if decision_obj.decision_details else None
    
    exec_settings = ExecutionSettings()
    history = TradeHistory(
        market_id=market.market_id,
        asset=market.asset,
        outcome_bought=decision,
        amount_usdc=0.0, # wait for fill
        executed_price=0.0,
        predicted_flip_prob=p_flip,
        p_up=decision_obj.p_up,
        strike=decision_obj.strike,
        active_features=_get_trade_active_features(asset_mode, active_features, decision_obj, market.asset),
        model_version=model_ver,
        model_key=model_key,
        confirm_model_key=confirm_model_key,
        confirm_model_version=confirm_model_version,
        model_attribution_source="EXACT" if model_key else None,
        status="PENDING",
        error_msg=None,
        mode=exec_settings.execution_mode.value,
        strategy_type=decision_obj.strategy_type,
        market_role=market_role,
        p_flip_effective=p_flip_effective,
        p_win_effective=decision_obj.p_win_effective,
        edge=round(edge, 4) if edge is not None else None,
        lgbm_metadata=lgbm_metadata,
        config_snapshot=config_snapshot_json,
        created_at=start_time,
        entry_filled_shares=0.0,
        entry_cost_usdc=0.0,
        remaining_shares=0.0,
        realized_pnl_usdc=0.0,
        position_status="OPENING"
    )
    
    if cfg.stop_loss_enabled:
        is_outsider = (
            hasattr(decision_obj, 'strategy_type')
            and isinstance(decision_obj.strategy_type, str)
            and decision_obj.strategy_type.upper() == "OUTSIDER"
        )
        stop_pct = cfg.stop_loss_pct_outsider if is_outsider else cfg.stop_loss_pct_favorite
        
        history.market_end_time = getattr(market, "end_time_est", None)
        history.stop_loss_pct = stop_pct
        # Price is unknown, will be set on fill
        history.stop_loss_status = "PENDING_FILL"

    if cfg.take_profit_enabled:
        history.take_profit_enabled    = True
        history.take_profit_multiplier = cfg.take_profit_multiplier
        history.take_profit_status     = "PENDING_FILL"
    else:
        history.take_profit_enabled = False
        history.take_profit_status  = "SKIPPED"

    savepoint = await db_session.begin_nested()
    try:
        db_session.add(history)
        await db_session.flush()

        result = await enqueue_open_request(
            db_session,
            trade_id=history.id,
            market_id=market.market_id,
            asset=market.asset,
            outcome_to_buy=decision,
            target_amount_usdc=actual_bet_size,
            limit_price=buy_price,
            requested_mode=exec_settings.execution_mode,
        )
        
        if result.disposition == EnqueueDisposition.BLOCKED:
            raise EnqueueRejected(f"Execution rejected: {result.reason}")

        if result.disposition == EnqueueDisposition.DUPLICATE:
            raise EnqueueRejected(f"ActiveExecutionConflict: Market {market.market_id} already has an active OPEN request.")

        from polyflip.db.execution_models import ExposureReservation
        from datetime import timedelta
        reservation = ExposureReservation(
            request_id=result.request_id,
            trade_history_id=history.id,
            market_id=market.market_id,
            amount_usdc=actual_bet_size,
            expires_at=start_time + timedelta(hours=1)
        )
        db_session.add(reservation)

        invalidate_stats_cache()
        invalidate_dashboard_cache()
    except Exception as e:
        await savepoint.rollback()
        logger.warning("enqueue_rejected_rolling_back", reason=str(e), market_id=market.market_id)
        raise
