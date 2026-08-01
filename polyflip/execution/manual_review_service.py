from dataclasses import dataclass
from decimal import Decimal
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
)


@dataclass(frozen=True)
class NoFillEligibility:
    allowed: bool
    blockers: tuple[str, ...]


async def evaluate_no_fill_eligibility(
    db: AsyncSession,
    request: ExecutionRequest,
) -> NoFillEligibility:
    blockers: list[str] = []

    if request.requested_mode != "LIVE":
        blockers.append("Заявка не относится к LIVE")

    if request.state != "MANUAL_REVIEW_REQUIRED":
        blockers.append(f"Некорректный статус: {request.state}")

    if Decimal(request.filled_shares or 0) != 0:
        blockers.append("Обнаружены заполненные shares")

    if Decimal(request.filled_cost_usdc or 0) != 0:
        blockers.append("Обнаружена стоимость исполнения")

    provider_evidence = await db.scalar(
        select(func.count(ExecutionAttempt.id)).where(
            ExecutionAttempt.request_id == request.id,
            or_(
                ExecutionAttempt.provider_order_id.is_not(None),
                ExecutionAttempt.tx_hash.is_not(None),
            ),
        )
    )

    if provider_evidence:
        blockers.append("Обнаружен provider_order_id или tx_hash")

    fill_count = await db.scalar(
        select(func.count(ExecutionFill.id))
        .join(
            ExecutionAttempt,
            ExecutionAttempt.id == ExecutionFill.attempt_id,
        )
        .where(ExecutionAttempt.request_id == request.id)
    )

    if fill_count:
        blockers.append("Обнаружены execution_fills")

    attempts = (
        await db.execute(
            select(ExecutionAttempt).where(
                ExecutionAttempt.request_id == request.id
            )
        )
    ).scalars().all()

    if any(a.provider_trade_ids for a in attempts):
        blockers.append("Обнаружены provider_trade_ids")

    if any(a.transaction_hashes for a in attempts):
        blockers.append("Обнаружены transaction_hashes")

    return NoFillEligibility(
        allowed=not blockers,
        blockers=tuple(blockers),
    )
