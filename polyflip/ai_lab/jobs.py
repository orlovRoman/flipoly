"""Restart-safe durable AI Lab job state transitions."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.service import utc_now
from polyflip.db.models import AIExperimentJob, AIRunStep

TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
CLAIMABLE_JOB_STATUSES = {"QUEUED", "RETRY_WAIT", "STALE"}


async def ensure_job(
    session: AsyncSession,
    *,
    run_id: int,
    step_id: int,
    operation: str,
    idempotency_key: str,
) -> AIExperimentJob:
    """Create one durable job for a run step, preserving idempotency."""
    row = (
        await session.execute(
            select(AIExperimentJob).where(
                AIExperimentJob.idempotency_key == idempotency_key
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = AIExperimentJob(
        run_id=run_id,
        step_id=step_id,
        operation=operation,
        idempotency_key=idempotency_key,
        status="QUEUED",
    )
    session.add(row)
    await session.flush()
    return row


async def claim_job(
    session: AsyncSession,
    idempotency_key: str,
) -> AIExperimentJob | None:
    row = (
        await session.execute(
            select(AIExperimentJob)
            .where(AIExperimentJob.idempotency_key == idempotency_key)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if row is None or row.status not in CLAIMABLE_JOB_STATUSES:
        return None
    now = utc_now()
    row.status = "RUNNING"
    row.attempt = int(row.attempt or 0) + 1
    row.started_at = row.started_at or now
    row.heartbeat_at = now
    row.error = None
    row.traceback = None
    await session.flush()
    return row


async def complete_job(
    session: AsyncSession,
    job_id: int,
    *,
    status: str,
    error: str | None = None,
    traceback: str | None = None,
) -> AIExperimentJob:
    row = await session.get(AIExperimentJob, job_id, with_for_update=True)
    if row is None:
        raise ValueError(f"job {job_id} not found")
    normalized = status.strip().upper()
    if normalized not in {"SUCCEEDED", "FAILED", "RETRY_WAIT", "CANCELLED"}:
        raise ValueError(f"unsupported terminal job status: {normalized}")
    row.status = normalized
    row.finished_at = utc_now() if normalized != "RETRY_WAIT" else None
    row.heartbeat_at = utc_now()
    row.error = error
    row.traceback = traceback
    await session.flush()
    return row


async def heartbeat_job(session: AsyncSession, job_id: int) -> AIExperimentJob:
    row = await session.get(AIExperimentJob, job_id, with_for_update=True)
    if row is None or row.status != "RUNNING":
        raise ValueError("job is not running")
    row.heartbeat_at = utc_now()
    await session.flush()
    return row


async def recover_stale_jobs(
    session: AsyncSession,
    *,
    stale_after_seconds: int = 300,
) -> int:
    cutoff = utc_now() - timedelta(seconds=stale_after_seconds)
    result = await session.execute(
        select(AIExperimentJob)
        .where(AIExperimentJob.status == "RUNNING")
        .where(AIExperimentJob.heartbeat_at < cutoff)
        .with_for_update()
    )
    rows = list(result.scalars().all())
    recovered = 0
    for row in rows:
        row.status = "STALE"
        row.error = "worker heartbeat expired"
        step = await session.get(AIRunStep, row.step_id, with_for_update=True)
        if step is not None and step.status == "RUNNING":
            step.status = "PENDING"
            step.finished_at = None
            step.error_code = None
            step.error_message = None
            recovered += 1
    await session.flush()
    return recovered
