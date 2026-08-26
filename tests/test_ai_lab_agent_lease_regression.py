"""Regression: lease atomicity and REQUEUE ownership (step 1, must fail before fix)."""

from __future__ import annotations

import asyncio

import pytest
import sqlalchemy as sa

from polyflip.api.ai_lab_agent import (
    AgentClaimRequest,
    CompleteRequest,
    claim_next_agent_run,
    complete_agent_run,
)
from polyflip.ai_lab.service import create_run
from polyflip.db.models import AIWorkerLease


async def _seed_queued(db_session) -> int:
    from polyflip.ai_lab.service import create_permission
    from uuid import uuid4

    perm = await create_permission(
        db_session,
        profile_name=f"reg-{uuid4().hex[:6]}",
        allowed_actions=["CREATE_EXPERIMENT"],
        scope={},
        limits={},
        updated_by="t",
        enabled=True,
    )
    run = await create_run(
        db_session,
        objective="x",
        scope={},
        autonomy_level="OBSERVE",
        budget_experiments=1,
        permission=perm,
        llm_provider="mock",
    )
    run.status = "QUEUED"
    await db_session.flush()
    await db_session.commit()
    return int(run.id)


@pytest.mark.asyncio
async def test_concurrent_claim_single_run_only_one_wins(db_session):
    run_id = await _seed_queued(db_session)
    first = await claim_next_agent_run(AgentClaimRequest(worker_id="w1"), db_session)
    second = await claim_next_agent_run(AgentClaimRequest(worker_id="w2"), db_session)
    successes = [r for r in (first, second) if r["run"] is not None]
    assert len(successes) == 1, f"expected exactly one winner, got {successes}"
    # the loser must not have created a second lease
    leases = (await db_session.execute(sa.select(AIWorkerLease))).scalars().all()
    assert len(leases) == 1


@pytest.mark.asyncio
async def test_expired_running_is_reclaimable(db_session):
    from datetime import datetime, timedelta, timezone

    run_id = await _seed_queued(db_session)
    first = await claim_next_agent_run(AgentClaimRequest(worker_id="w1"), db_session)
    assert first["run"] is not None
    # expire the lease
    lease = (
        await db_session.execute(
            sa.select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
        )
    ).scalar_one()
    lease.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await db_session.commit()
    second = await claim_next_agent_run(AgentClaimRequest(worker_id="w2"), db_session)
    assert second["run"] is not None
    assert second["run"]["id"] == run_id


@pytest.mark.asyncio
async def test_requeue_without_token_rejected(db_session):
    run_id = await _seed_queued(db_session)
    claimed = (
        await claim_next_agent_run(AgentClaimRequest(worker_id="w1"), db_session)
    )["run"]
    # try to requeue without lease_token - should be 409, not 200
    with pytest.raises(Exception) as exc:
        await complete_agent_run(
            run_id, CompleteRequest(action="REQUEUE", reason="x"), db_session
        )
    assert (
        "409" in str(exc.value)
        or "LEASE" in str(exc.value)
        or "lease" in str(exc.value).lower()
    )


@pytest.mark.asyncio
async def test_requeue_with_foreign_token_rejected(db_session):
    run_id = await _seed_queued(db_session)
    claimed = (
        await claim_next_agent_run(AgentClaimRequest(worker_id="w1"), db_session)
    )["run"]
    with pytest.raises(Exception):
        await complete_agent_run(
            run_id,
            CompleteRequest(action="REQUEUE", reason="x", lease_token="wrong-token"),
            db_session,
        )


@pytest.mark.asyncio
async def test_postgres_concurrent_claim_has_exactly_one_winner(
    pg_session_factory,
):
    """Two PostgreSQL transactions must not claim the same run."""
    async with pg_session_factory() as setup_session:
        run_id = await _seed_queued(setup_session)
        run = await setup_session.get(
            __import__(
                "polyflip.db.models", fromlist=["AIOptimizationRun"]
            ).AIOptimizationRun,
            run_id,
        )
        permission_id = int(run.permission_id)

    async def attempt(worker_id: str):
        async with pg_session_factory() as session:
            return await claim_next_agent_run(
                AgentClaimRequest(worker_id=worker_id),
                session,
            )

    try:
        first, second = await asyncio.gather(
            attempt("postgres-worker-1"),
            attempt("postgres-worker-2"),
        )
        winners = [item["run"] for item in (first, second) if item["run"]]
        assert len(winners) == 1
        assert winners[0]["id"] == run_id

        async with pg_session_factory() as verify_session:
            leases = (
                (
                    await verify_session.execute(
                        sa.select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
                    )
                )
                .scalars()
                .all()
            )
            assert len(leases) == 1
    finally:
        from polyflip.db.models import AIOptimizationRun, AIPermission

        async with pg_session_factory() as cleanup_session:
            run = await cleanup_session.get(AIOptimizationRun, run_id)
            if run is not None:
                await cleanup_session.delete(run)
            permission = await cleanup_session.get(AIPermission, permission_id)
            if permission is not None:
                await cleanup_session.delete(permission)
            await cleanup_session.commit()
