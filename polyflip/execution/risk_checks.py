from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.execution.config import ExecutionMode
from polyflip.execution.exposure import get_reserved_exposure
from polyflip.execution.states import ACTIVE_POSITION_STATES
from decimal import Decimal
import datetime
from uuid import UUID


async def check_risk_limits(
    session: AsyncSession,
    intent: str,
    max_spend_usdc: Decimal,
    requested_mode: str,
    request_id: UUID | None = None,
) -> str | None:
    """
    Проверяет риск-лимиты до отправки ордера.
    Возвращает строку с описанием нарушения, или None если всё ОК.

    CLOSE-заявки всегда проходят — они уменьшают экспозицию.
    """
    if intent == "CLOSE":
        return None

    if requested_mode == "LIVE":
        rt_stmt = select(RuntimeSettings).where(
            RuntimeSettings.key == "LIVE_TRADING_ENABLED"
        )
        rt_set = (await session.execute(rt_stmt)).scalar_one_or_none()
        if not rt_set or rt_set.value.lower() != "true":
            return "LIVE trading kill switch is off"

    # --- MAX_SINGLE_ORDER_USDC ---
    single_limit_stmt = select(RuntimeSettings).where(
        RuntimeSettings.key == "MAX_SINGLE_ORDER_USDC"
    )
    single_limit_set = (await session.execute(single_limit_stmt)).scalar_one_or_none()
    if single_limit_set:
        try:
            single_limit = Decimal(single_limit_set.value)
            if max_spend_usdc > single_limit:
                return (
                    f"Single order size {max_spend_usdc} USDC exceeds limit {single_limit} USDC"
                )
        except (ValueError, TypeError) as exc:
            return f"Invalid MAX_SINGLE_ORDER_USDC configuration: {exc}"

    # --- MAX_OPEN_POSITIONS ---
    # Считаем все активные позиции в данном режиме (OPEN + PARTIALLY_CLOSED + EXIT_REQUESTED)
    max_open_stmt = select(RuntimeSettings).where(
        RuntimeSettings.key == "MAX_OPEN_POSITIONS"
    )
    max_open_set = (await session.execute(max_open_stmt)).scalar_one_or_none()
    if max_open_set:
        try:
            max_open = int(max_open_set.value)
            open_count_stmt = select(func.count(TradeHistory.id)).where(
                TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
                TradeHistory.mode == requested_mode,
            )
            open_count = (await session.execute(open_count_stmt)).scalar_one()
            if open_count >= max_open:
                return f"Max open positions limit reached ({max_open})"
        except (ValueError, TypeError) as exc:
            return f"Invalid MAX_OPEN_POSITIONS configuration: {exc}"

    # --- MAX_TOTAL_EXPOSURE_USDC ---
    # Для частично закрытых позиций экспозиция = entry_cost * remaining / entry_filled
    max_exp_stmt = select(RuntimeSettings).where(
        RuntimeSettings.key == "MAX_TOTAL_EXPOSURE_USDC"
    )
    max_exp_set = (await session.execute(max_exp_stmt)).scalar_one_or_none()
    if max_exp_set:
        try:
            max_exp = Decimal(max_exp_set.value)

            # Взвешенная экспозиция: для каждой позиции берём оставшуюся стоимость
            remaining_cost_expr = case(
                (
                    TradeHistory.entry_filled_shares > 0,
                    TradeHistory.entry_cost_usdc
                    * TradeHistory.remaining_shares
                    / TradeHistory.entry_filled_shares,
                ),
                else_=Decimal("0"),
            )

            current_exp_stmt = select(
                func.coalesce(func.sum(remaining_cost_expr), Decimal("0"))
            ).where(
                TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
                TradeHistory.mode == requested_mode,
            )
            current_exp = (await session.execute(current_exp_stmt)).scalar_one()

            # Зарезервированная экспозиция по активным заявкам
            res_exp = await get_reserved_exposure(
                session,
                mode=ExecutionMode(requested_mode),
                exclude_request_id=request_id,
            )

            total = Decimal(str(current_exp or 0)) + res_exp + max_spend_usdc
            if total > max_exp:
                return (
                    f"Max total exposure {max_exp} USDC would be exceeded. "
                    f"Current: {current_exp}, Reserved: {res_exp}, Requested: {max_spend_usdc}"
                )
        except (ValueError, TypeError) as exc:
            return f"Invalid MAX_TOTAL_EXPOSURE_USDC configuration: {exc}"

    # --- DAILY_LOSS_LIMIT_USDC ---
    today_start = datetime.datetime.now(datetime.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    loss_limit_stmt = select(RuntimeSettings).where(
        RuntimeSettings.key == "DAILY_LOSS_LIMIT_USDC"
    )
    loss_limit_set = (await session.execute(loss_limit_stmt)).scalar_one_or_none()
    if loss_limit_set:
        try:
            loss_limit = Decimal(loss_limit_set.value)

            daily_pnl_stmt = select(
                func.coalesce(func.sum(TradeHistory.realized_pnl_usdc), Decimal("0"))
            ).where(
                TradeHistory.mode == requested_mode,
                TradeHistory.position_status == "CLOSED",
                TradeHistory.closed_at >= today_start,
                TradeHistory.realized_pnl_usdc.is_not(None),
            )
            daily_pnl = (await session.execute(daily_pnl_stmt)).scalar_one()

            # loss_limit — положительное число; убыток — отрицательный PnL
            if Decimal(str(daily_pnl or 0)) <= -loss_limit:
                return (
                    f"Daily loss limit {loss_limit} USDC reached. "
                    f"Today PnL: {daily_pnl}"
                )
        except (ValueError, TypeError) as exc:
            return f"Invalid DAILY_LOSS_LIMIT_USDC configuration: {exc}"

    return None
