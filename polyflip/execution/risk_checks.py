from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.db.execution_models import ExposureReservation
from decimal import Decimal

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
            open_count_stmt = select(func.count(TradeHistory.id)).where(TradeHistory.position_status == "OPEN")
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
            current_exp_stmt = select(func.sum(TradeHistory.entry_cost_usdc)).where(TradeHistory.position_status == "OPEN")
            current_exp_res = await session.execute(current_exp_stmt)
            current_exp = current_exp_res.scalar_one_or_none() or Decimal("0")
            
            # Plus reserved exposure, excluding current request
            res_exp_stmt = select(func.sum(ExposureReservation.amount_usdc)).where(ExposureReservation.released_at.is_(None))
            if request_id:
                res_exp_stmt = res_exp_stmt.where(ExposureReservation.request_id != request_id)
            res_exp_res = await session.execute(res_exp_stmt)
            res_exp = res_exp_res.scalar_one_or_none() or Decimal("0")
            
            if Decimal(str(current_exp)) + Decimal(str(res_exp)) + max_spend_usdc > max_exp:
                return f"Max total exposure limit reached ({max_exp} USDC)"
        except (ValueError, TypeError) as exc:
            return f"Invalid risk configuration: {exc}"

    return None
