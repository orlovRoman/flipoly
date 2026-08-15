"""Autonomous background daemon runner for AI Lab Agent (Phase 10).

Acquires and maintains leases, executes researcher iterations, and handles graceful shutdown.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.ai_lab.agent import AILabAgent
from polyflip.ai_lab.service import utc_now
from polyflip.db.connection import async_session
from polyflip.db.models import AIOptimizationRun, AIWorkerLease

logger = structlog.get_logger("polyflip.ai_lab.agent_runner")

LEASE_TTL_SECONDS = 120
HEARTBEAT_INTERVAL_SECONDS = 30
POLL_INTERVAL_SECONDS = 10


class AgentRunner:
    """Background worker that continuously claims and advances autonomous optimization runs."""

    def __init__(self) -> None:
        self.worker_id = f"ai-agent-{uuid.uuid4().hex[:8]}"
        self.running = True

    async def acquire_lease(self, session: AsyncSession, run_id: int) -> bool:
        """Attempt to acquire or renew an exclusive worker lease for an optimization run."""
        now = utc_now()
        stmt = select(AIWorkerLease).where(AIWorkerLease.run_id == run_id).with_for_update()
        lease = (await session.execute(stmt)).scalar_one_or_none()

        if lease is None:
            new_lease = AIWorkerLease(
                run_id=run_id,
                owner_token=self.worker_id,
                acquired_at=now,
                heartbeat_at=now,
                expires_at=now + timedelta(seconds=LEASE_TTL_SECONDS),
            )
            session.add(new_lease)
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
        """Release the worker lease upon step completion or termination."""
        stmt = select(AIWorkerLease).where(
            AIWorkerLease.run_id == run_id,
            AIWorkerLease.owner_token == self.worker_id,
        )
        lease = (await session.execute(stmt)).scalar_one_or_none()
        if lease:
            await session.delete(lease)
            await session.commit()

    async def process_one_run(self) -> bool:
        """Find one eligible run and advance it by one iteration."""
        async with async_session() as session:
            stmt = (
                select(AIOptimizationRun)
                .where(AIOptimizationRun.status.in_(["QUEUED", "RUNNING", "DRAFT"]))
                .order_by(desc(AIOptimizationRun.id))
                .limit(1)
            )
            run = (await session.execute(stmt)).scalar_one_or_none()
            if run is None:
                return False

            run_id = run.id

        async with async_session() as session:
            has_lease = await self.acquire_lease(session, run_id)
            if not has_lease:
                logger.info("lease_unavailable", run_id=run_id, worker=self.worker_id)
                return False

            try:
                agent = AILabAgent(session)
                result = await agent.execute_iteration(run_id)
                logger.info("iteration_completed", run_id=run_id, result=result)
                return True
            except Exception as exc:
                logger.error("iteration_failed", run_id=run_id, error=str(exc))
                return False
            finally:
                await self.release_lease(session, run_id)

    async def run_loop(self) -> None:
        """Main execution loop for the autonomous runner."""
        logger.info("agent_runner_started", worker=self.worker_id)

        while self.running:
            try:
                progressed = await self.process_one_run()
                if not progressed:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logger.info("agent_runner_cancelled")
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
            pass  # Windows support

    await runner.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
