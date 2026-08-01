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
        (
            await db.execute(
                select(ExecutionAttempt).where(
                    ExecutionAttempt.request_id == request.id
                )
            )
        )
        .scalars()
        .all()
    )

    if any(a.provider_trade_ids for a in attempts):
        blockers.append("Обнаружены provider_trade_ids")

    if any(a.transaction_hashes for a in attempts):
        blockers.append("Обнаружены transaction_hashes")

    return NoFillEligibility(
        allowed=not blockers,
        blockers=tuple(blockers),
    )


async def evaluate_no_fill_eligibility_batch(
    db: AsyncSession,
    requests: list[ExecutionRequest],
) -> dict[uuid.UUID, NoFillEligibility]:
    import uuid
    from typing import Dict
    from collections import defaultdict

    if not requests:
        return {}

    req_map = {req.id: req for req in requests}
    req_ids = list(req_map.keys())

    provider_evidence_counts = dict(
        (
            await db.execute(
                select(ExecutionAttempt.request_id, func.count(ExecutionAttempt.id))
                .where(
                    ExecutionAttempt.request_id.in_(req_ids),
                    or_(
                        ExecutionAttempt.provider_order_id.is_not(None),
                        ExecutionAttempt.tx_hash.is_not(None),
                    ),
                )
                .group_by(ExecutionAttempt.request_id)
            )
        ).all()
    )

    fill_counts = dict(
        (
            await db.execute(
                select(ExecutionAttempt.request_id, func.count(ExecutionFill.id))
                .join(
                    ExecutionAttempt,
                    ExecutionAttempt.id == ExecutionFill.attempt_id,
                )
                .where(ExecutionAttempt.request_id.in_(req_ids))
                .group_by(ExecutionAttempt.request_id)
            )
        ).all()
    )

    attempts = (
        (
            await db.execute(
                select(ExecutionAttempt).where(ExecutionAttempt.request_id.in_(req_ids))
            )
        )
        .scalars()
        .all()
    )

    attempts_by_req = defaultdict(list)
    for attempt in attempts:
        attempts_by_req[attempt.request_id].append(attempt)

    results = {}
    for req in requests:
        blockers: list[str] = []

        if req.requested_mode != "LIVE":
            blockers.append("Заявка не относится к LIVE")

        if req.state != "MANUAL_REVIEW_REQUIRED":
            blockers.append(f"Некорректный статус: {req.state}")

        if Decimal(req.filled_shares or 0) != 0:
            blockers.append("Обнаружены заполненные shares")

        if Decimal(req.filled_cost_usdc or 0) != 0:
            blockers.append("Обнаружена стоимость исполнения")

        if provider_evidence_counts.get(req.id, 0) > 0:
            blockers.append("Обнаружен provider_order_id или tx_hash")

        if fill_counts.get(req.id, 0) > 0:
            blockers.append("Обнаружены execution_fills")

        req_attempts = attempts_by_req.get(req.id, [])

        if any(a.provider_trade_ids for a in req_attempts):
            blockers.append("Обнаружены provider_trade_ids")

        if any(a.transaction_hashes for a in req_attempts):
            blockers.append("Обнаружены transaction_hashes")

        results[req.id] = NoFillEligibility(
            allowed=not blockers,
            blockers=tuple(blockers),
        )

    return results
