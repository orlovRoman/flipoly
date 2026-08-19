"""Restart-safe durable AI Lab job state transitions."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.service import utc_now
from polyflip.db.models import AIExperimentJob

TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}


async def claim_job(session: AsyncSession, idempotency_key: str) -> AIExperimentJob | None:
    row = (
        await session.execute(
            select(AIExperimentJob)
            .where(AIExperimentJob.idempotency_key == idempotency_key)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if row is None or row.status in TERMINAL_JOB_STATUSES:
        return None
    now = utc_now()
    row.status = "RUNNING"
    row.attempt = int(row.attempt or 0) + 1
    row.started_at = row.started_at or now
    row.heartbeat_at = now
    await session.flush()
    return row


async def heartbeat_job(session: AsyncSession, job_id: int) -> AIExperimentJob:
    row = await session.get(AIExperimentJob, job_id, with_for_update=True)
    if row is None or row.status != "RUNNING":
        raise ValueError("job is not running")
    row.heartbeat_at = utc_now()
    await session.flush()
    return row


async def recover_stale_jobs(session: AsyncSession, *, stale_after_seconds: int = 300) -> int:
    cutoff = utc_now() - timedelta(seconds=stale_after_seconds)
    result = await session.execute(
        select(AIExperimentJob)
        .where(AIExperimentJob.status == "RUNNING")
        .where(AIExperimentJob.heartbeat_at < cutoff)
        .with_for_update()
    )
    rows = list(result.scalars().all())
    for row in rows:
        row.status = "STALE"
    await session.flush()
    return len(rows)
