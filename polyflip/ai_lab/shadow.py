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
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        existing = (
            await session.execute(
                select(AIShadowObservation).where(
                    AIShadowObservation.idempotency_key == key
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            raise
        return existing
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


async def record_decision_shadow(
    session: AsyncSession,
    *,
    asset: str,
    market_id: str,
    snapshot_at: datetime,
    run_id: int | None,
    active_model_key: str | None,
    candidate_model_key: str | None,
    active_action: str | None,
    candidate_action: str | None,
    active_probability: float | None,
    candidate_probability: float | None,
    candidate_ask: float | None,
    active_net_edge: float | None,
    candidate_net_edge: float | None,
    lr_direction_vote: str | None,
    lgbm_direction_vote: str | None,
    consensus_type: str | None,
) -> AIShadowObservation | None:
    """Persist one passive same-snapshot candidate-vs-active observation.

    Missing assignments or candidate model keys are normal and return None.
    This helper never changes an execution decision; callers should catch
    unexpected persistence errors at the decision boundary.
    """
    normalized = str(asset or "").strip().upper()
    assignment = (
        await session.execute(
            select(AIShadowAssignment)
            .where(
                AIShadowAssignment.asset.in_(
                    [normalized, normalized.removesuffix("USDT") + "USDT"]
                ),
                AIShadowAssignment.status == "RUNNING",
            )
            .order_by(AIShadowAssignment.created_at.desc(), AIShadowAssignment.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if assignment is None or not candidate_model_key:
        return None
    return await record_shadow_observation(
        session,
        assignment_id=int(assignment.id),
        run_id=run_id or assignment.run_id,
        market_id=str(market_id),
        snapshot_at=snapshot_at,
        values={
            "active_model_key": active_model_key,
            "candidate_model_key": candidate_model_key,
            "active_action": active_action,
            "candidate_action": candidate_action,
            "active_probability": active_probability,
            "candidate_probability": candidate_probability,
            "candidate_ask": candidate_ask,
            "active_net_edge": active_net_edge,
            "candidate_net_edge": candidate_net_edge,
            "lr_direction_vote": lr_direction_vote,
            "lgbm_direction_vote": lgbm_direction_vote,
            "consensus_type": consensus_type,
            "shadow_logreg_action": active_action,
            "actual_combined_action": active_action,
            "shadow_logreg_net_edge": active_net_edge,
            "actual_net_edge": active_net_edge,
        },
    )
