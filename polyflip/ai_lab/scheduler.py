"""Bounded, auditable scheduler for offline AI Lab LightGBM work.

The scheduler is intentionally a finite invocation, not a daemon. A caller
(cron, an operator, or an AI agent) starts one bounded cycle; the cycle acquires
a database lease, executes short worker batches, and releases the lease. It
never activates models or touches live execution.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.lgbm_worker import (
    MAX_LGBM_WORKER_STEPS,
    execute_lgbm_steps,
)
from polyflip.ai_lab.executor import ExecutionBatchError, ExecutionOutcome
from polyflip.ai_lab.service import utc_now
from polyflip.db.models import AIOptimizationRun, AIWorkerLease

MAX_SCHEDULER_ITERATIONS = 20
MAX_SCHEDULER_INTERVAL_SECONDS = 60.0
MIN_LEASE_TTL_SECONDS = 30.0
MAX_LEASE_TTL_SECONDS = 3600.0


@dataclass(frozen=True)
class SchedulerResult:
    """Serializable summary of one bounded scheduler invocation."""

    status: str
    run_id: int
    owner_token: str
    iterations: int
    outcomes: tuple[ExecutionOutcome, ...]
    stop_reason: str


def validate_scheduler_limits(
    *,
    max_iterations: int,
    max_steps: int,
    interval_seconds: float,
    lease_ttl_seconds: float,
) -> None:
    if not 1 <= max_iterations <= MAX_SCHEDULER_ITERATIONS:
        raise ValueError(
            f"max_iterations must be between 1 and {MAX_SCHEDULER_ITERATIONS}"
        )
    if not 1 <= max_steps <= MAX_LGBM_WORKER_STEPS:
        raise ValueError(
            f"max_steps must be between 1 and {MAX_LGBM_WORKER_STEPS}"
        )
    if not 0 <= interval_seconds <= MAX_SCHEDULER_INTERVAL_SECONDS:
        raise ValueError(
            "interval_seconds must be between 0 and "
            f"{MAX_SCHEDULER_INTERVAL_SECONDS:g}"
        )
    if not MIN_LEASE_TTL_SECONDS <= lease_ttl_seconds <= MAX_LEASE_TTL_SECONDS:
        raise ValueError(
            "lease_ttl_seconds must be between "
            f"{MIN_LEASE_TTL_SECONDS:g} and {MAX_LEASE_TTL_SECONDS:g}"
        )


async def acquire_worker_lease(
    session: AsyncSession,
    run_id: int,
    owner_token: str,
    *,
    ttl_seconds: float,
) -> bool:
    """Acquire or renew a lease; return False when another owner is active."""
    now = utc_now()
    row = (
        await session.execute(
            select(AIWorkerLease)
            .where(AIWorkerLease.run_id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if row is not None and row.expires_at > now and row.owner_token != owner_token:
        await session.rollback()
        return False

    if row is None:
        row = AIWorkerLease(
            run_id=run_id,
            owner_token=owner_token,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        session.add(row)
    else:
        row.owner_token = owner_token
        row.heartbeat_at = now
        row.expires_at = now + timedelta(seconds=ttl_seconds)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return False
    return True


async def renew_worker_lease(
    session: AsyncSession,
    run_id: int,
    owner_token: str,
    *,
    ttl_seconds: float,
) -> bool:
    row = (
        await session.execute(
            select(AIWorkerLease)
            .where(AIWorkerLease.run_id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if row is None or row.owner_token != owner_token:
        await session.rollback()
        return False
    now = utc_now()
    row.heartbeat_at = now
    row.expires_at = now + timedelta(seconds=ttl_seconds)
    await session.commit()
    return True


async def release_worker_lease(
    session: AsyncSession,
    run_id: int,
    owner_token: str,
) -> None:
    await session.execute(
        delete(AIWorkerLease).where(
            AIWorkerLease.run_id == run_id,
            AIWorkerLease.owner_token == owner_token,
        )
    )
    await session.commit()


async def run_lgbm_scheduler(
    session: AsyncSession,
    run_id: int,
    *,
    max_iterations: int = 1,
    max_steps: int = 1,
    interval_seconds: float = 0.0,
    lease_ttl_seconds: float = 120.0,
    owner_token: str | None = None,
) -> SchedulerResult:
    """Run a finite, leased sequence of offline worker batches.

    The loop stops when there are no pending steps, when a batch returns a
    non-success outcome, or when the configured iteration bound is reached.
    """
    validate_scheduler_limits(
        max_iterations=max_iterations,
        max_steps=max_steps,
        interval_seconds=interval_seconds,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise ValueError(f"AI Lab run {run_id} not found")
    if run.status not in {"PLANNING", "RUNNING"}:
        raise ValueError(f"run {run_id} cannot execute from {run.status}")

    token = owner_token or uuid.uuid4().hex
    acquired = await acquire_worker_lease(
        session,
        run_id,
        token,
        ttl_seconds=lease_ttl_seconds,
    )
    if not acquired:
        return SchedulerResult(
            status="ALREADY_RUNNING",
            run_id=run_id,
            owner_token=token,
            iterations=0,
            outcomes=(),
            stop_reason="another worker lease is active",
        )

    outcomes: list[ExecutionOutcome] = []
    stop_reason = "iteration_limit"
    status = "COMPLETED"
    iterations = 0
    try:
        for iteration in range(max_iterations):
            iterations = iteration + 1
            batch = await execute_lgbm_steps(
                session,
                run_id,
                max_steps=max_steps,
            )
            outcomes.extend(batch)
            if not batch:
                stop_reason = "no_pending_steps"
                break
            if any(item.status != "SUCCEEDED" for item in batch):
                status = "STOPPED_ON_OUTCOME"
                stop_reason = "non_success_outcome"
                break
            if not await renew_worker_lease(
                session,
                run_id,
                token,
                ttl_seconds=lease_ttl_seconds,
            ):
                status = "LEASE_LOST"
                stop_reason = "lease_lost"
                break
            if interval_seconds and iteration + 1 < max_iterations:
                await asyncio.sleep(interval_seconds)
    except ExecutionBatchError as exc:
        outcomes.extend(exc.completed)
        status = "PARTIAL_FAILURE"
        stop_reason = "batch_failure"
    except Exception:
        status = "FAILED"
        stop_reason = "scheduler_exception"
        raise
    finally:
        # A failed adapter may leave the session in an invalid transaction
        # state. Reset it before DELETE+COMMIT in lease release.
        try:
            await session.rollback()
        finally:
            await release_worker_lease(session, run_id, token)

    return SchedulerResult(
        status=status,
        run_id=run_id,
        owner_token=token,
        iterations=iterations,
        outcomes=tuple(outcomes),
        stop_reason=stop_reason,
    )
