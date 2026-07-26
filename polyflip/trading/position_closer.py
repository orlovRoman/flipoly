import structlog
from datetime import datetime, timezone
import asyncio
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_
from uuid import uuid4, UUID
from decimal import Decimal

from polyflip.db.models import TradeHistory, LiveMarket, SlippageLog
from polyflip.trading.trader import PolyTrader
from polyflip.trading.schemas import TradeExecution, ExecutionFees

logger = structlog.get_logger(__name__)

DUST_SHARES = Decimal("0.01")

async def claim_position_for_exit(
    db_session: AsyncSession, 
    trade_id: int, 
    reason: str, 
    stale_before: datetime
) -> Optional[UUID]:
    now = datetime.now(timezone.utc)
    attempt_id = uuid4()

    claimable = or_(
        TradeHistory.position_status.in_(
            ["OPEN", "EXIT_FAILED", "PARTIALLY_CLOSED"]
        ),
        and_(
            TradeHistory.position_status == "CLOSING",
            TradeHistory.exit_claimed_at < stale_before,
        ),
    )

    stmt = (
        update(TradeHistory)
        .where(
            TradeHistory.id == trade_id,
            claimable,
        )
        .values(
            position_status="CLOSING",
            exit_reason=reason,
            exit_attempt_id=attempt_id,
            exit_claimed_at=now,
            exit_attempts=TradeHistory.exit_attempts + 1,
            updated_at=now,
        )
        .returning(TradeHistory.id)
    )

    claimed = (await db_session.execute(stmt)).scalar_one_or_none()
    if claimed is None:
        await db_session.rollback()
        return None

    await db_session.commit()
    return attempt_id

async def _fetch_order_status(trader: PolyTrader, order_id: str, execution: TradeExecution) -> TradeExecution:
    """Опрашивает CLOB API и обновляет TradeExecution реальными данными исполнения."""
    try:
        await asyncio.sleep(1.0)
        order_info = await trader.get_order(order_id)
        if not order_info:
            import dataclasses
            return dataclasses.replace(execution, error_message="Order not found in CLOB")
            
        filled_str = order_info.get("sizeMatched")
        price_str = order_info.get("priceMatched")
        status = order_info.get("status", "LIVE")
        
        filled = Decimal(str(filled_str)) if filled_str is not None else Decimal("0")
        avg_price = Decimal(str(price_str)) if price_str is not None else execution.average_price
        
        import dataclasses
        exec_status = "UNKNOWN"
        if status == "LIVE":
            exec_status = "LIVE"
            if filled > 0:
                exec_status = "PARTIALLY_FILLED"
        elif status == "MATCHED":
            exec_status = "FILLED"
        elif status == "CANCELED" or status == "CANCELLED":
            exec_status = "CANCELLED"
            if filled > 0:
                exec_status = "PARTIALLY_FILLED"
        elif status == "EXPIRED":
            exec_status = "REJECTED"
        
        fees = execution.fees
        platform_fee = Decimal("0") # could parse from API
        # Polymarket taker fee is usually shares * feeRate * price * (1-price)
        
        return dataclasses.replace(
            execution,
            status=exec_status,
            provider_status=status,
            filled_shares=filled,
            average_price=avg_price,
            net_position_delta_shares=filled if execution.side == "BUY" else -filled,
            fees=fees,
            observed_at=datetime.now(timezone.utc)
        )
    except Exception as e:
        logger.error("fetch_order_status_error", order_id=order_id, error=str(e))
        import dataclasses
        return dataclasses.replace(execution, error_message=f"fetch_order_error: {e}")

async def execute_position_exit(
    db_session: AsyncSession,
    trader: PolyTrader,
    trade: TradeHistory,
    market: LiveMarket,
    target_price: float,
    fee_rate: float,
    attempt_id: UUID,
    actual_size: Optional[float] = None
):
    token_to_sell = market.yes_token_id if trade.outcome_bought == "YES" else market.no_token_id
    
    # Calculate exact size from remaining_shares if available, else fallback to old calculation
    if trade.remaining_shares is not None:
        size_to_sell = float(trade.remaining_shares)
    else:
        size_to_sell = actual_size if actual_size is not None else (trade.amount_usdc / trade.executed_price if trade.executed_price > 0 else 0)
        
    execution = await trader.execute_trade(
        market_id=market.market_id,
        token_id=token_to_sell,
        side="SELL",
        price=target_price,
        size=size_to_sell
    )
    
    # Override attempt_id to match our database claim
    import dataclasses
    execution = dataclasses.replace(execution, attempt_id=attempt_id)
    
    if execution.status == "UNKNOWN" and execution.provider_order_id:
        execution = await _fetch_order_status(trader, execution.provider_order_id, execution)
        
    await reconcile_position_exit(
        db_session, trader, trade.id, execution, target_price, fee_rate, trade.market_id, trade.asset, trade.outcome_bought
    )

async def reconcile_position_exit(
    db_session: AsyncSession, 
    trader: PolyTrader,
    trade_id: int, 
    execution: TradeExecution,
    target_price: float,
    fee_rate: float,
    market_id: str,
    asset: str,
    outcome_bought: str
):
    now = datetime.now(timezone.utc)
    
    trade = (
        await db_session.execute(
            select(TradeHistory)
            .where(
                TradeHistory.id == trade_id,
                TradeHistory.exit_attempt_id == execution.attempt_id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    
    if trade is None:
        return
        
    if execution.status == "UNKNOWN" or execution.status == "LIVE":
        trade.position_status = "CLOSING"
        trade.last_exit_error = execution.error_message or "unknown or pending execution state"
        trade.updated_at = now
        await db_session.commit()
        return
        
    filled = min(execution.filled_shares, trade.remaining_shares or execution.original_requested_shares)
    
    if filled <= 0:
        trade.position_status = "EXIT_FAILED"
        trade.last_exit_error = execution.error_message
        trade.updated_at = now
        await db_session.commit()
        return
        
    # Calculate PnL
    avg_price = execution.average_price if execution.average_price is not None else Decimal(str(target_price))
    fee_dec = Decimal(str(fee_rate))
    gross_proceeds = avg_price * filled
    net_proceeds = gross_proceeds * (Decimal("1") - fee_dec)
    
    if execution.fees.platform_fee_usdc is not None:
        net_proceeds -= execution.fees.platform_fee_usdc
    if execution.fees.builder_fee_usdc is not None:
        net_proceeds -= execution.fees.builder_fee_usdc
        
    entry_price = Decimal(str(trade.executed_price))
    cost_basis = entry_price * filled
    fill_realized_pnl = net_proceeds - cost_basis
    
    if execution.fees.network_fee_usdc is not None:
        fill_realized_pnl -= execution.fees.network_fee_usdc

    # Apply updates
    if trade.realized_pnl_usdc is None:
        trade.realized_pnl_usdc = Decimal("0")
    if trade.remaining_shares is None:
        # Initial fixup for legacy data
        trade.remaining_shares = Decimal(str(trade.amount_usdc / trade.executed_price if trade.executed_price > 0 else 0))
        
    trade.realized_pnl_usdc += fill_realized_pnl
    trade.remaining_shares -= filled
    
    # Old PNL accumulation for compatibility
    if trade.pnl is None:
        trade.pnl = float(fill_realized_pnl)
    else:
        trade.pnl += float(fill_realized_pnl)

    if trade.remaining_shares <= DUST_SHARES:
        trade.remaining_shares = Decimal("0")
        trade.position_status = "CLOSED"
        trade.closed_at = now
    else:
        trade.position_status = "PARTIALLY_CLOSED"
        trade.closed_at = None
        
    trade.close_price = float(avg_price)
    
    if trade.exit_reason == "TAKE_PROFIT":
        trade.take_profit_sell_size = (trade.take_profit_sell_size or Decimal("0")) + filled
    elif trade.exit_reason == "STOP_LOSS":
        trade.stop_loss_sell_size = (trade.stop_loss_sell_size or Decimal("0")) + filled
        
    trade.exit_order_id = execution.provider_order_id or ""
    trade.updated_at = now
    
    # Log slippage
    slip = Decimal(str(target_price)) - avg_price
    slip_pct = float((slip / Decimal(str(target_price)) * 100) if target_price > 0 else 0.0)
    slip_cost = float(slip * filled)
    
    db_session.add(SlippageLog(
        trade_id=trade.id,
        market_id=market_id,
        asset=asset,
        outcome_bought=outcome_bought,
        expected_price=target_price,
        executed_price=float(avg_price),
        slippage=float(slip),
        slippage_pct=slip_pct,
        bet_size_usdc=trade.amount_usdc,
        slippage_cost_usdc=slip_cost,
        mode="PAPER" if execution.status == "PAPER_FILLED" else "LIVE",
        created_at=now,
    ))

    await db_session.commit()
