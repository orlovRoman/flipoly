from decimal import Decimal
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.execution_models import ExposureReservation, ExecutionRequest
from polyflip.execution.states import ACTIVE_REQUEST_STATES
from polyflip.execution.config import ExecutionMode

async def get_reserved_exposure(
    session: AsyncSession,
    *,
    mode: ExecutionMode,
    market_id: str | None = None,
    exclude_request_id: UUID | None = None,
) -> Decimal:
    stmt = (
        select(
            func.coalesce(
                func.sum(ExposureReservation.amount_usdc), Decimal("0")
            )
        )
        .join(
            ExecutionRequest,
            ExecutionRequest.id == ExposureReservation.request_id,
        )
        .where(
            ExposureReservation.released_at.is_(None),
            ExecutionRequest.requested_mode == mode.value,
            ExecutionRequest.state.in_(ACTIVE_REQUEST_STATES),
        )
    )

    if market_id is not None:
        stmt = stmt.where(
            ExposureReservation.market_id == market_id
        )

    if exclude_request_id is not None:
        stmt = stmt.where(
            ExposureReservation.request_id != exclude_request_id
        )

    value = (await session.execute(stmt)).scalar_one()
    return Decimal(str(value))
