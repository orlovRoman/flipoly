"""Restart-safe durable AI Lab job state transitions."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.service import utc_now
from polyflip.db.models import AIExperimentJob, AIRunStep

TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
CLAIMABLE_JOB_STATUSES = {"QUEUED", "RETRY_WAIT", "STALE"}
MAX_RETRY_ATTEMPTS = 3
RETRYABLE_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "temporary",
    "429",
    "502",
    "503",
    "504",
    "rate limit",
    "deadlock",
    "too many connections",
    "service unavailable",
)


def is_retryable_error(error: BaseException | str | None) -> bool:
    """Return whether an adapter failure is safe to retry automatically.

    This intentionally uses a conservative marker list.  Unknown failures
    remain permanent so malformed data or code errors are not retried forever.
    """
    if error is None:
        return False
    text = str(error).strip().lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def should_retry(error: BaseException | str | None, attempt: int) -> bool:
    """Return whether a transient failure still has retry budget."""
    return attempt < MAX_RETRY_ATTEMPTS and is_retryable_error(error)


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
    try:
        # A uniqueness race must not roll back the caller's transaction
        # (claim_next_step may already have marked the step RUNNING). Keep
        # the speculative insert inside a savepoint and re-read the winner.
        async with session.begin_nested():
            session.add(row)
            await session.flush()
    except IntegrityError:
        row = (
            await session.execute(
                select(AIExperimentJob).where(
                    AIExperimentJob.idempotency_key == idempotency_key
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise
    return row


async def claim_job(
    session: AsyncSession,
    idempotency_key: str,
    *,
    owner_token: str | None = None,
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
    row.owner_token = owner_token
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
    owner_token: str | None = None,
) -> AIExperimentJob:
    row = await session.get(AIExperimentJob, job_id, with_for_update=True)
    if row is None:
        raise ValueError(f"job {job_id} not found")
    if owner_token is not None and row.owner_token not in {None, owner_token}:
        raise ValueError("job is owned by another worker")
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


async def heartbeat_job(
    session: AsyncSession,
    job_id: int,
    *,
    owner_token: str | None = None,
) -> AIExperimentJob:
    row = await session.get(AIExperimentJob, job_id, with_for_update=True)
    if row is None or row.status != "RUNNING":
        raise ValueError("job is not running")
    if owner_token is not None and row.owner_token != owner_token:
        raise ValueError("job is owned by another worker")
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
        step = await session.get(
            AIRunStep, getattr(row, "step_id", None), with_for_update=True
        )
        if step is not None and step.status == "RUNNING":
            step.status = "PENDING"
            step.finished_at = None
            step.error_code = None
            step.error_message = None
            recovered += 1
    await session.flush()
    return recovered
