from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.execution.config import ExecutionMode
from polyflip.execution.exposure import get_reserved_exposure
from decimal import Decimal
import datetime
from uuid import UUID

async def check_risk_limits(session: AsyncSession, intent: str, max_spend_usdc: Decimal, requested_mode: str, request_id: UUID | None = None) -> str | None:
    """
    Checks risk limits before order submission. 
    Returns an error string if a limit is breached, otherwise None.
    """
    if intent == "CLOSE":
        # Always allow CLOSE orders to reduce exposure
        return None
        
    if requested_mode == "LIVE":
        # Check kill switch again just in case
        rt_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
        rt_res = await session.execute(rt_stmt)
        rt_set = rt_res.scalar_one_or_none()
        if not rt_set or rt_set.value.lower() != "true":
            return "LIVE trading kill switch is off"
            
    # Check max open positions
    max_open_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MAX_OPEN_POSITIONS")
    max_open_res = await session.execute(max_open_stmt)
    max_open_set = max_open_res.scalar_one_or_none()
    
    if max_open_set:
        try:
            max_open = int(max_open_set.value)
            # Count currently OPEN positions
            open_count_stmt = select(func.count(TradeHistory.id)).where(
                TradeHistory.position_status == "OPEN",
                TradeHistory.mode == requested_mode
            )
            open_count = (await session.execute(open_count_stmt)).scalar_one()
            if open_count >= max_open:
                return f"Max open positions limit reached ({max_open})"
        except (ValueError, TypeError) as exc:
            return f"Invalid risk configuration: {exc}"

    # Check total exposure
    max_exp_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MAX_TOTAL_EXPOSURE_USDC")
    max_exp_res = await session.execute(max_exp_stmt)
    max_exp_set = max_exp_res.scalar_one_or_none()
    
    if max_exp_set:
        try:
            max_exp = Decimal(max_exp_set.value)
            # Current exposure in TradeHistory
            current_exp_stmt = select(func.coalesce(func.sum(TradeHistory.entry_cost_usdc), Decimal("0"))).where(
                TradeHistory.position_status == "OPEN",
                TradeHistory.mode == requested_mode
            )
            current_exp = (await session.execute(current_exp_stmt)).scalar_one()
            
            # Plus reserved exposure, excluding current request
            res_exp = await get_reserved_exposure(
                session, 
                mode=ExecutionMode(requested_mode), 
                exclude_request_id=request_id
            )
            
            if current_exp + res_exp + max_spend_usdc > max_exp:
                return f"Max total exposure limit reached ({max_exp} USDC). Current: {current_exp}, Reserved: {res_exp}, Requested: {max_spend_usdc}"
        except (ValueError, TypeError) as exc:
            return f"Invalid risk configuration: {exc}"

    # Daily loss limit
    today_start = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    loss_limit_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "DAILY_LOSS_LIMIT_USDC")
    loss_limit_res = await session.execute(loss_limit_stmt)
    loss_limit_set = loss_limit_res.scalar_one_or_none()
    if loss_limit_set:
        try:
            limit = Decimal(loss_limit_set.value)
            if limit >= 0:
                return "Invalid DAILY_LOSS_LIMIT_USDC: expected a negative number"

            stmt = select(
                func.coalesce(func.sum(TradeHistory.realized_pnl_usdc), Decimal("0"))
            ).where(
                TradeHistory.mode == requested_mode,
                TradeHistory.position_status == "CLOSED",
                func.coalesce(
                    TradeHistory.closed_at,
                    TradeHistory.updated_at,
                    TradeHistory.created_at,
                ) >= today_start,
            )

            daily_pnl = Decimal(str((await session.execute(stmt)).scalar_one()))

            if daily_pnl <= limit:
                return f"Daily loss limit breached (Limit: {limit}, Loss: {daily_pnl})"
        except (ValueError, TypeError):
            pass

    return None
