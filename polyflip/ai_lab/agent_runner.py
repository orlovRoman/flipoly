"""Durable autonomous AI Lab agent worker.

The API only queues runs. This process claims them, renews its lease while
training/OOT executes, and persists failures with traceback.
"""
from __future__ import annotations

import asyncio
import signal
import traceback
import uuid
from datetime import timedelta

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.ai_lab.agent import AILabAgent
from polyflip.ai_lab.agent_tools import expire_overlays
from polyflip.ai_lab.jobs import recover_stale_jobs
from polyflip.ai_lab.service import transition_run, utc_now
from polyflip.db.connection import async_session
from polyflip.db.models import AIExperimentJob, AIOptimizationRun, AIWorkerLease

logger = structlog.get_logger("polyflip.ai_lab.agent_runner")

LEASE_TTL_SECONDS = 120
HEARTBEAT_INTERVAL_SECONDS = 30
POLL_INTERVAL_SECONDS = 10


class AgentRunner:
    def __init__(self) -> None:
        self.worker_id = f"ai-agent-{uuid.uuid4().hex[:8]}"
        self.running = True

    async def acquire_lease(self, session: AsyncSession, run_id: int) -> bool:
        now = utc_now()
        lease = (
            await session.execute(
                select(AIWorkerLease)
                .where(AIWorkerLease.run_id == run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if lease is None:
            session.add(
                AIWorkerLease(
                    run_id=run_id,
                    owner_token=self.worker_id,
                    acquired_at=now,
                    heartbeat_at=now,
                    expires_at=now + timedelta(seconds=LEASE_TTL_SECONDS),
                )
            )
            await session.commit()
            return True
        if lease.owner_token == self.worker_id or lease.expires_at < now:
            lease.owner_token = self.worker_id
            lease.heartbeat_at = now
            lease.expires_at = now + timedelta(seconds=LEASE_TTL_SECONDS)
            await session.commit()
            return True
        return False

    async def release_lease(self, session: AsyncSession, run_id: int) -> None:
        lease = (
            await session.execute(
                select(AIWorkerLease).where(
                    AIWorkerLease.run_id == run_id,
                    AIWorkerLease.owner_token == self.worker_id,
                )
            )
        ).scalar_one_or_none()
        if lease is not None:
            await session.delete(lease)
            await session.commit()

    async def _heartbeat_loop(self, run_id: int) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            async with async_session() as heartbeat_session:
                if not await self.acquire_lease(heartbeat_session, run_id):
                    logger.error("ai_worker_lease_lost", run_id=run_id)
                    return
                jobs = (
                    await heartbeat_session.execute(
                        select(AIExperimentJob).where(
                            AIExperimentJob.run_id == run_id,
                            AIExperimentJob.status == "RUNNING",
                        )
                    )
                ).scalars().all()
                now = utc_now()
                for job in jobs:
                    job.heartbeat_at = now
                await heartbeat_session.commit()

    async def _mark_failed(self, run_id: int, exc: BaseException) -> None:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        async with async_session() as failed_session:
            run = (
                await failed_session.execute(
                    select(AIOptimizationRun)
                    .where(AIOptimizationRun.id == run_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if run is None or run.status in {
                "COMPLETED", "CANCELLED", "REJECTED", "ROLLED_BACK",
                "INSUFFICIENT_DATA", "FAILED",
            }:
                return
            run.error = error_text[:12000]
            try:
                await transition_run(
                    failed_session, run, "FAILED", reason="agent worker exception"
                )
            except Exception:
                run.status = "FAILED"
                run.finished_at = utc_now()
            await failed_session.commit()

    async def process_one_run(self) -> bool:
        async with async_session() as session:
            await expire_overlays(session)
            await recover_stale_jobs(session)
            await session.commit()
            run = (
                await session.execute(
                    select(AIOptimizationRun)
                    .where(
                        AIOptimizationRun.status.in_(
                            ["QUEUED", "RUNNING", "EVALUATING"]
                        )
                    )
                    .order_by(desc(AIOptimizationRun.id))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if run is None:
                return False
            run_id = run.id

        async with async_session() as session:
            if not await self.acquire_lease(session, run_id):
                logger.info("lease_unavailable", run_id=run_id, worker=self.worker_id)
                return False
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(run_id))
            try:
                result = await AILabAgent(session).execute_iteration(run_id)
                logger.info("iteration_completed", run_id=run_id, result=result)
                return True
            except Exception as exc:
                await session.rollback()
                logger.error("iteration_failed", run_id=run_id, error=str(exc))
                await self._mark_failed(run_id, exc)
                return False
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                await session.rollback()
                try:
                    await self.release_lease(session, run_id)
                except Exception:
                    await session.rollback()

    async def run_loop(self) -> None:
        logger.info("agent_runner_started", worker=self.worker_id)
        while self.running:
            try:
                progressed = await self.process_one_run()
                if not progressed:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("agent_runner_loop_error", error=str(exc))
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        logger.info("agent_runner_stopped", worker=self.worker_id)

    def stop(self) -> None:
        self.running = False


async def main() -> None:
    runner = AgentRunner()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, runner.stop)
        except NotImplementedError:
            pass
    await runner.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
