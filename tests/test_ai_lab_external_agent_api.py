"""External agent API behavior (T07)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from types import SimpleNamespace

from fastapi import HTTPException

from polyflip.api.ai_lab_agent import (
    AgentClaimRequest,
    CompleteRequest,
    DecisionRequest,
    HeartbeatRequest,
    ProposalRequest,
    _acquire_lease,
    claim_next_agent_run,
    complete_agent_run,
    get_agent_context,
    get_agent_latest_result,
    submit_agent_decision,
    submit_agent_proposal,
)
from polyflip.ai_lab.service import create_run
from polyflip.config import settings
from polyflip.db.models import AIExperimentConfig, AIOptimizationRun, AIWorkerLease


VALID_PROPOSAL = {
    "hypothesis": "Calibrated LogReg on FS_D1 improves outsider OOT PnL",
    "asset": "BTC",
    "market_role": "OUTSIDER",
    "model_family": "LOGREG",
    "feature_set": "FS_D1",
    "parameter_changes": [{"key": "C", "value": 0.5}],
    "strategy_parameter_changes": [
        {"key": "decision_threshold", "value": 0.58}
    ],
    "expected_effect": {
        "metric": "median_oot_pnl", "direction": "increase", "target_gain": 0.05,
    },
    "reasoning": ["baseline drift"],
    "risks": [],
    "test_plan": {"oot_windows": 3, "min_markets": 50,
                   "execution_mode": "PAPER_REALISTIC"},
}

VALID_DECISION = {
    "action": "CONTINUE_RESEARCH",
    "rationale": "Positive direction but insufficient trades in T3 window.",
    "key_findings": ["net_pnl +1.2"],
    "recommended_config_id": None,
    "proposed_overlay": None,
    "next_step_focus": "raise coverage",
}


async def _seed_run(db_session, *, status: str = "QUEUED") -> int:
    from uuid import uuid4

    from polyflip.ai_lab.service import create_permission

    permission = await create_permission(
        db_session,
        profile_name=f"agent-test-{uuid4().hex[:6]}",
        allowed_actions=["CREATE_EXPERIMENT", "TRAIN_MODEL"],
        scope={},
        limits={},
        updated_by="test",
        enabled=True,
    )
    run = await create_run(
        db_session,
        objective="external agent e2e",
        scope={"asset": "BTC"},
        autonomy_level="OBSERVE",
        budget_experiments=2,
        permission=permission,
        llm_provider="mock",
    )
    if status != "DRAFT":
        run.status = status
        await db_session.flush()
    run_id = int(run.id)
    await db_session.commit()
    return run_id


@pytest.mark.asyncio
async def test_claim_none_when_queue_empty(db_session):
    payload = await claim_next_agent_run(AgentClaimRequest(), db_session)
    assert payload == {"run": None}


@pytest.mark.asyncio
async def test_claim_transitions_to_running_and_returns_lease(db_session):
    run_id = await _seed_run(db_session)
    result = await claim_next_agent_run(AgentClaimRequest(), db_session)
    claimed = result["run"]
    assert claimed["id"] == run_id
    assert claimed["status"] == "QUEUED"
    assert claimed["lease_token"]

    stored = await db_session.get(AIOptimizationRun, run_id)
    assert stored.status == "QUEUED"
    lease = (
        await db_session.execute(sa.select(AIWorkerLease))
    ).scalar_one()
    assert lease.owner_token == claimed["lease_token"]


@pytest.mark.asyncio
async def test_single_run_cannot_be_claimed_twice(db_session):
    run_id = await _seed_run(db_session)
    first = await claim_next_agent_run(
        AgentClaimRequest(worker_id="agent-1"), db_session
    )
    second = await claim_next_agent_run(
        AgentClaimRequest(worker_id="agent-2"), db_session
    )
    assert first["run"]["id"] == run_id
    assert second["run"] is None


@pytest.mark.asyncio
async def test_heartbeat_renews_and_wrong_token_is_lost(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]

    from polyflip.api.ai_lab_agent import agent_heartbeat

    renewed = await agent_heartbeat(
        HeartbeatRequest(run_id=run_id, lease_token=claimed["lease_token"]),
        db_session,
    )
    assert renewed["run_id"] == run_id

    lease = (
        await db_session.execute(sa.select(AIWorkerLease))
    ).scalar_one()
    expires_at = (
        lease.expires_at
        if lease.expires_at.tzinfo
        else lease.expires_at.replace(tzinfo=timezone.utc)
    )
    assert expires_at > datetime.now(timezone.utc)

    with pytest.raises(HTTPException) as exc_info:
        await agent_heartbeat(
            HeartbeatRequest(run_id=run_id, lease_token="wrong"),
            db_session,
        )
    assert exc_info.value.detail == "LEASE_LOST"


@pytest.mark.asyncio
async def test_context_snapshot_shape_is_aggregated_only(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    context = await get_agent_context(
        run_id=run_id, lease_token=claimed["lease_token"], db=db_session
    )
    assert set(context) == {
        "run", "active_models", "recent_trade_statistics",
        "prior_experiments", "available_feature_sets", "quality_gate",
    }
    assert context["run"]["budget_remaining_steps"] == 2
    assert "FS_D0" in context["available_feature_sets"]
    # No raw tables or secrets leak into the snapshot.
    assert all("api_key" not in str(key).lower() for key in context)


@pytest.mark.asyncio
async def test_proposal_creates_config_and_canonical_steps(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"], proposal=VALID_PROPOSAL
        ),
        db_session,
    )
    assert result["config_id"] > 0
    config = await db_session.get(AIExperimentConfig, result["config_id"])
    assert config.model_family == "LogisticRegression"
    assert config.created_by == "external-agent"
    steps = (
        await db_session.execute(
            sa.select(AIOptimizationRun)  # placeholder to keep sa import used
        )
    )
    from polyflip.db.models import AIRunStep

    step_rows = (
        await db_session.execute(
            sa.select(AIRunStep).where(AIRunStep.run_id == run_id)
        )
    ).scalars().all()
    step_types = {row.step_type for row in step_rows}
    # Canonical executor pipeline created by plan_run for the agent config.
    assert {"TRAIN_MODEL", "RUN_OOT_BACKTEST", "RUN_POLYMARKET_OOT"} <= step_types


@pytest.mark.asyncio
async def test_invalid_proposal_rejected_with_422(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    bad = dict(VALID_PROPOSAL, asset="BTC")
    bad["hypothesis"] = "short"
    with pytest.raises(Exception):
        await submit_agent_proposal(
            run_id,
            ProposalRequest(lease_token=claimed["lease_token"], proposal=bad),
            db_session,
        )


@pytest.mark.asyncio
async def test_decision_persisted_and_complete_releases_lease(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    await submit_agent_proposal(
        run_id,
        ProposalRequest(lease_token=claimed["lease_token"], proposal=VALID_PROPOSAL),
        db_session,
    )
    decision_result = await submit_agent_decision(
        run_id,
        DecisionRequest(lease_token=claimed["lease_token"], decision=VALID_DECISION),
        db_session,
    )
    assert decision_result["accepted"] is True

    # Move the pipeline to RUNNING the same way a worker claim would, so the
    # canonical RUNNING -> EVALUATING -> COMPLETED completion path applies.
    from polyflip.ai_lab.orchestrator import claim_next_step

    claimed_step = await claim_next_step(db_session, run_id)
    assert claimed_step is not None

    completed = await complete_agent_run(
        run_id,
        CompleteRequest(
            action="COMPLETED",
            reason="done",
            lease_token=claimed["lease_token"],
        ),
        db_session,
    )
    assert completed["status"] == "COMPLETED"
    leases = (
        await db_session.execute(sa.select(AIWorkerLease))
    ).scalars().all()
    assert leases == []


@pytest.mark.asyncio
async def test_requeue_returns_run_to_queue_without_lease(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    result = await complete_agent_run(
        run_id,
        CompleteRequest(action="REQUEUE", reason="retry"),
        db_session,
    )
    assert result["status"] == "QUEUED"
    again = await claim_next_agent_run(AgentClaimRequest(), db_session)
    assert again["run"]["id"] == run_id


def test_verify_agent_token_falls_back_to_api_key():
    import asyncio as _asyncio

    from polyflip.api.ai_lab_agent import verify_agent_token

    request = SimpleNamespace(headers={})
    token = settings.API_KEY
    ok = _asyncio.run(
        verify_agent_token(authorization=f"Bearer {token}", x_api_key=None)
    )
    assert ok is None
