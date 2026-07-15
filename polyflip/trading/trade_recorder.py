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
import os
import structlog
from polyflip.constants import TRADING_MODE_CRYPTO, TRADING_MODE_ML, TRADING_MODE_FAVORITE

logger = structlog.get_logger(__name__)

def _get_trade_active_features(asset_mode: str, active_features_str: str, decision_obj: Any) -> str:
    if asset_mode == TRADING_MODE_CRYPTO:
        return "CRYPTO_TREND"
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
    active_features: str = ""
):
    """Сохраняет запись о пропуске сделки в БД или обновляет её причину."""
    if existing_skipped:
        if (existing_skipped.error_msg != reason or 
            existing_skipped.predicted_flip_prob != p_flip_val or 
            existing_skipped.edge != edge or
            existing_skipped.active_features != active_features):
            existing_skipped.error_msg = reason
            existing_skipped.predicted_flip_prob = p_flip_val
            existing_skipped.model_version = model_version
            existing_skipped.edge = edge
            if active_features:
                existing_skipped.active_features = active_features
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
            mode="LIVE" if bool(os.getenv("POLYGON_PRIVATE_KEY") and os.getenv("POLYGON_ADDRESS")) else "PAPER",
            edge=edge,
            created_at=start_time
        )
        db_session.add(history)



async def execute_and_record(
    db_session: AsyncSession,
    trader: Any,
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
) -> None:
    if not validation.valid:
        await save_or_update_skipped_trade(
            db_session, market, validation.skip_reason, p_flip, model_ver, start_time,
            existing_skipped=existing_skipped,
            edge=validation.edge,
            active_features=_get_trade_active_features(asset_mode, active_features, decision_obj)
        )
        return
        
    decision = decision_obj.action.replace("BUY_", "")
    buy_price = validation.buy_price
    actual_bet_size = validation.actual_bet_size
    edge = validation.edge
    num_shares = round(actual_bet_size / buy_price, 2)
    token_to_buy = market.yes_token_id if decision == "YES" else market.no_token_id
    
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

    try:
        trade_res = await trader.execute_trade(
            market_id=market.market_id,
            token_id=token_to_buy,
            side="BUY",
            price=buy_price,
            size=num_shares
        )
    except Exception as e:
        logger.exception("execute_trade_error", error=str(e))
        trade_res = {"status": "FAILED", "error_msg": str(e), "executed_price": buy_price, "executed_usdc": actual_bet_size, "mode": "PAPER"}

    if existing_skipped:
        await db_session.delete(existing_skipped)

    history = TradeHistory(
        market_id=market.market_id,
        asset=market.asset,
        outcome_bought=decision,
        amount_usdc=trade_res.get("executed_usdc", actual_bet_size),
        executed_price=trade_res.get("executed_price", buy_price),
        predicted_flip_prob=p_flip,
        p_up=decision_obj.p_up,
        strike=decision_obj.strike,
        active_features=_get_trade_active_features(asset_mode, active_features, decision_obj),
        model_version=model_ver,
        status=trade_res.get("status", "FAILED"),
        error_msg=trade_res.get("error_msg"),
        mode=trade_res.get("mode", "PAPER"),
        edge=round(edge, 4) if edge is not None else None,
        created_at=start_time
    )
    db_session.add(history)
    await db_session.flush()

    if trade_res.get("status") == "SUCCESS":
        exec_p = trade_res.get("executed_price", buy_price)
        slip = round(exec_p - buy_price, 6)
        slip_pct = round(slip / buy_price * 100, 4) if buy_price > 0 else 0.0
        slip_cost = round(slip * (actual_bet_size / exec_p), 4) if exec_p > 0 else 0.0

        slippage_record = SlippageLog(
            trade_id=history.id,
            market_id=market.market_id,
            asset=market.asset,
            outcome_bought=decision,
            expected_price=buy_price,
            executed_price=exec_p,
            slippage=slip,
            slippage_pct=slip_pct,
            bet_size_usdc=actual_bet_size,
            slippage_cost_usdc=slip_cost,
            mode=trade_res.get("mode", "PAPER"),
            created_at=start_time,
        )
        db_session.add(slippage_record)
        
        if cfg.stop_loss_enabled:
            is_outsider = (
                hasattr(decision_obj, 'strategy_type')
                and isinstance(decision_obj.strategy_type, str)
                and decision_obj.strategy_type.upper() == "OUTSIDER"
            )
            stop_pct = cfg.stop_loss_pct_outsider if is_outsider else cfg.stop_loss_pct_favorite
            
            history.market_end_time = getattr(market, "end_time_est", None)
            history.stop_loss_pct = stop_pct
            history.stop_loss_price = compute_stop_price(exec_p, stop_pct)
            history.stop_loss_status = "ACTIVE"

        if cfg.take_profit_enabled:
            history.take_profit_enabled    = True
            history.take_profit_multiplier = cfg.take_profit_multiplier
            history.take_profit_price      = compute_take_profit_price(exec_p, cfg.take_profit_multiplier)
            history.take_profit_status     = "ACTIVE"
        else:
            history.take_profit_enabled = False
            history.take_profit_status  = "SKIPPED"

    invalidate_stats_cache()
    invalidate_dashboard_cache()
