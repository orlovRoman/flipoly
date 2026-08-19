"""Persistence helpers for same-snapshot passive shadow observations."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import AIShadowObservation


def observation_key(assignment_id: int, market_id: str, snapshot_at: datetime) -> str:
    ts = snapshot_at.astimezone(timezone.utc).isoformat()
    return f"shadow:{int(assignment_id)}:{market_id}:{ts}"


async def record_shadow_observation(
    session: AsyncSession,
    *,
    assignment_id: int,
    run_id: int | None,
    market_id: str,
    snapshot_at: datetime,
    values: Mapping[str, Any],
) -> AIShadowObservation:
    """Insert one observation or return the existing idempotent row."""
    key = observation_key(assignment_id, market_id, snapshot_at)
    existing = (
        await session.execute(
            select(AIShadowObservation).where(AIShadowObservation.idempotency_key == key)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    allowed = {column.name for column in AIShadowObservation.__table__.columns}
    payload = {key: value for key, value in values.items() if key in allowed}
    row = AIShadowObservation(
        assignment_id=assignment_id,
        run_id=run_id,
        market_id=market_id,
        snapshot_at=snapshot_at,
        idempotency_key=key,
        **payload,
    )
    session.add(row)
    await session.flush()
    return row


async def resolve_shadow_observation(
    session: AsyncSession,
    observation_id: int,
    *,
    market_outcome: str,
    active_pnl: float | None,
    candidate_pnl: float | None,
) -> AIShadowObservation:
    row = await session.get(AIShadowObservation, observation_id, with_for_update=True)
    if row is None:
        raise ValueError(f"shadow observation {observation_id} not found")
    if row.status == "RESOLVED":
        return row
    row.market_outcome = market_outcome
    row.active_pnl = active_pnl
    row.candidate_pnl = candidate_pnl
    row.status = "RESOLVED"
    row.resolved_at = datetime.now(timezone.utc)
    await session.flush()
    return row
