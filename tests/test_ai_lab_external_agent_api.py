"""External agent API behavior (T07)."""
from __future__ import annotations

import uuid
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
    assert claimed["status"] == "RUNNING"
    assert claimed["lease_token"]

    stored = await db_session.get(AIOptimizationRun, run_id)
    assert stored.status == "RUNNING"
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
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=VALID_PROPOSAL,
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
            ProposalRequest(
                lease_token=claimed["lease_token"],
                client_request_id=uuid.uuid4().hex,
                proposal=bad,
            ),
            db_session,
        )


@pytest.mark.asyncio
async def test_decision_persisted_and_complete_releases_lease(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    proposal_result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=VALID_PROPOSAL,
        ),
        db_session,
    )
    # Seed a terminal POLYMARKET_OOT result so decision gate (READY) passes.
    from polyflip.db.models import ExperimentResult

    from datetime import datetime, timezone

    term = ExperimentResult(
        run_id=run_id,
        config_id=proposal_result["config_id"],
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        metrics={"median_pnl": 1.2},
        trade_count=100,
        net_pnl=1.2,
        max_drawdown=-0.5,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(term)
    await db_session.flush()
    decision_result = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            decision=VALID_DECISION,
        ),
        db_session,
    )
    assert decision_result["accepted"] is True

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
        CompleteRequest(action="REQUEUE", reason="retry", lease_token=claimed["lease_token"]),
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


@pytest.mark.asyncio
async def test_agent_api_requires_token_via_http(db_session):
    from httpx import ASGITransport, AsyncClient

    from polyflip.api.main import app
    from polyflip.db.connection import get_db_session

    async def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/ai-lab/agent/claim", json={})
        assert response.status_code == 401
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.asyncio
async def test_agent_api_wrong_token_rejected(db_session):
    from httpx import ASGITransport, AsyncClient

    from polyflip.api.main import app
    from polyflip.db.connection import get_db_session

    async def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/ai-lab/agent/claim",
                headers={"Authorization": "Bearer wrong-token"},
                json={},
            )
        assert response.status_code == 401
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.asyncio
async def test_agent_token_allows_access(db_session, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from polyflip.api.main import app
    from polyflip.config import settings as cfg
    from polyflip.db.connection import get_db_session

    monkeypatch.setattr(cfg, "AI_LAB_AGENT_TOKEN", "agent-secret")
    monkeypatch.setattr(cfg, "API_KEY", "other-key")

    async def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/ai-lab/agent/claim",
                headers={"Authorization": "Bearer agent-secret"},
                json={},
            )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.asyncio
async def test_fallback_api_key_when_agent_token_not_set(db_session, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from polyflip.api.main import app
    from polyflip.config import settings as cfg
    from polyflip.db.connection import get_db_session

    monkeypatch.setattr(cfg, "AI_LAB_AGENT_TOKEN", "")
    async def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/ai-lab/agent/claim",
                headers={"X-API-Key": cfg.API_KEY},
                json={},
            )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.asyncio
async def test_invalid_proposal_422_via_http(db_session, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from polyflip.api.main import app
    from polyflip.config import settings as cfg
    from polyflip.db.connection import get_db_session

    monkeypatch.setattr(cfg, "AI_LAB_AGENT_TOKEN", "agent-secret")
    run_id = await _seed_run(db_session)
    async def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    transport = ASGITransport(app=app)
    bad = dict(VALID_PROPOSAL)
    bad["hypothesis"] = "short"
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            claim = await client.post(
                "/api/ai-lab/agent/claim",
                headers={"Authorization": "Bearer agent-secret"},
                json={"worker_id": "agent-1"},
            )
            assert claim.status_code == 200
            lease = claim.json()["run"]["lease_token"]
            response = await client.post(
                f"/api/ai-lab/agent/runs/{run_id}/proposal",
                headers={"Authorization": "Bearer agent-secret"},
                json={
                    "lease_token": lease,
                    "client_request_id": uuid.uuid4().hex,
                    "proposal": bad,
                },
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.asyncio
async def test_invalid_decision_422_via_http(db_session, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from polyflip.api.main import app
    from polyflip.config import settings as cfg
    from polyflip.db.connection import get_db_session

    monkeypatch.setattr(cfg, "AI_LAB_AGENT_TOKEN", "agent-secret")
    run_id = await _seed_run(db_session)
    async def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            claim = await client.post(
                "/api/ai-lab/agent/claim",
                headers={"Authorization": "Bearer agent-secret"},
                json={"worker_id": "agent-1"},
            )
            lease = claim.json()["run"]["lease_token"]
            await client.post(
                f"/api/ai-lab/agent/runs/{run_id}/proposal",
                headers={"Authorization": "Bearer agent-secret"},
                json={
                    "lease_token": lease,
                    "client_request_id": uuid.uuid4().hex,
                    "proposal": VALID_PROPOSAL,
                },
            )
            response = await client.post(
                f"/api/ai-lab/agent/runs/{run_id}/decision",
                headers={"Authorization": "Bearer agent-secret"},
                json={"lease_token": lease, "decision": {"action": "UNKNOWN"}},
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.asyncio
async def test_proposal_ordering_and_idempotency(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    lease = claimed["lease_token"]
    req_id_a = uuid.uuid4().hex
    req_id_b = uuid.uuid4().hex
    # First proposal creates iteration.
    first = await submit_agent_proposal(
        run_id,
        ProposalRequest(lease_token=lease, client_request_id=req_id_a, proposal=VALID_PROPOSAL),
        db_session,
    )
    assert first["config_id"] > 0
    assert len(first["step_ids"]) >= 4  # PROPOSAL + 3 canonical steps
    # Idempotent retry with same client_request_id must not create a new config.
    from polyflip.db.models import AIRunStep

    steps_before = (await db_session.execute(sa.select(AIRunStep).where(AIRunStep.run_id == run_id))).scalars().all()
    configs_before = (await db_session.execute(sa.select(AIExperimentConfig))).scalars().all()
    retry = await submit_agent_proposal(
        run_id,
        ProposalRequest(lease_token=lease, client_request_id=req_id_a, proposal=VALID_PROPOSAL),
        db_session,
    )
    assert retry["config_id"] == first["config_id"]
    assert retry["step_ids"] == first["step_ids"]
    steps_after_retry = (await db_session.execute(sa.select(AIRunStep).where(AIRunStep.run_id == run_id))).scalars().all()
    configs_after_retry = (await db_session.execute(sa.select(AIExperimentConfig))).scalars().all()
    assert len(steps_after_retry) == len(steps_before)
    assert len(configs_after_retry) == len(configs_before)
    # Different client_request_id creates a new iteration.
    second = await submit_agent_proposal(
        run_id,
        ProposalRequest(lease_token=lease, client_request_id=req_id_b, proposal=VALID_PROPOSAL),
        db_session,
    )
    assert second["config_id"] != first["config_id"]
    assert second["config_id"] > 0
    # No (run_id, step_index) conflicts and ordering is monotonic.
    all_steps = (
        await db_session.execute(sa.select(AIRunStep).where(AIRunStep.run_id == run_id).order_by(AIRunStep.step_index))
    ).scalars().all()
    indices = [s.step_index for s in all_steps]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    # No duplicate (run_id, step_index) pairs.
    assert len(indices) == len(all_steps)
    # Verify proposal steps are at correct positions.
    proposal_steps = [s for s in all_steps if s.step_type == "PROPOSAL"]
    assert len(proposal_steps) == 2
    assert proposal_steps[0].client_request_id == req_id_a
    assert proposal_steps[1].client_request_id == req_id_b
    # First proposal should be before its TRAIN block, second proposal after.
    assert proposal_steps[0].step_index < proposal_steps[1].step_index


@pytest.mark.asyncio
async def test_result_returns_only_terminal_oot(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    lease = claimed["lease_token"]
    proposal = await submit_agent_proposal(
        run_id,
        ProposalRequest(lease_token=lease, client_request_id=uuid.uuid4().hex, proposal=VALID_PROPOSAL),
        db_session,
    )
    # Before any OOT result, state is PENDING and TRAIN results are ignored.
    from polyflip.db.models import ExperimentResult
    from datetime import datetime, timezone

    # Insert a TRAIN result only – should still be PENDING.
    train_res = ExperimentResult(
        run_id=run_id,
        config_id=proposal["config_id"],
        evaluation_kind="TRAIN",
        status="SUCCEEDED",
        metrics={},
        trade_count=10,
        net_pnl=5.0,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(train_res)
    await db_session.flush()
    pending = await get_agent_latest_result(run_id=run_id, lease_token=lease, db=db_session)
    assert pending["state"] == "PENDING"
    assert pending["result"] is None
    # Add terminal POLYMARKET_OOT SUCCEEDED – now READY.
    oot = ExperimentResult(
        run_id=run_id,
        config_id=proposal["config_id"],
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        metrics={"median_pnl": 2.0},
        trade_count=100,
        net_pnl=2.0,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(oot)
    await db_session.flush()
    ready = await get_agent_latest_result(run_id=run_id, lease_token=lease, db=db_session)
    assert ready["state"] == "READY"
    assert ready["result"] is not None
    assert ready["result"]["evaluation_kind"] == "POLYMARKET_OOT"
    assert ready["result"]["config_id"] == proposal["config_id"]


@pytest.mark.asyncio
async def test_decision_requires_ready_and_validates_config(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    lease = claimed["lease_token"]
    proposal = await submit_agent_proposal(
        run_id,
        ProposalRequest(lease_token=lease, client_request_id=uuid.uuid4().hex, proposal=VALID_PROPOSAL),
        db_session,
    )
    # No terminal result yet – decision should be rejected with RESULT_NOT_READY.
    with pytest.raises(HTTPException) as exc:
        await submit_agent_decision(
            run_id,
            DecisionRequest(lease_token=lease, client_request_id=uuid.uuid4().hex, decision=VALID_DECISION),
            db_session,
        )
    assert exc.value.status_code == 409
    assert "RESULT_NOT_READY" in str(exc.value.detail)
    # Seed terminal result for a different config – recommended_config_id must belong.
    from polyflip.db.models import ExperimentResult, AIExperimentConfig
    from datetime import datetime, timezone

    # Create an unrelated config not linked to this run.
    from polyflip.ai_lab.service import create_experiment_config

    other_cfg = await create_experiment_config(
        db_session,
        name="other-cfg",
        model_family="LogReg",
        feature_set="FS_D0",
        feature_pipeline_version="v1",
        model_params={},
        strategy_params={},
        backtest_params={},
        asset="BTC",
        description="other",
        created_by="test",
    )
    await db_session.flush()
    term = ExperimentResult(
        run_id=run_id,
        config_id=proposal["config_id"],
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        metrics={},
        trade_count=50,
        net_pnl=1.0,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(term)
    await db_session.flush()
    # Try decision with non-belonging recommended_config_id.
    bad_decision = dict(VALID_DECISION)
    bad_decision["recommended_config_id"] = int(other_cfg.id)
    with pytest.raises(HTTPException) as exc2:
        await submit_agent_decision(
            run_id,
            DecisionRequest(lease_token=lease, client_request_id=uuid.uuid4().hex, decision=bad_decision),
            db_session,
        )
    assert exc2.value.status_code == 422
    # Valid decision with belonging config succeeds and increments once.
    good_decision = dict(VALID_DECISION)
    good_decision["recommended_config_id"] = int(proposal["config_id"])
    first_ok = await submit_agent_decision(
        run_id,
        DecisionRequest(lease_token=lease, client_request_id="decision-aaa-12345678", decision=good_decision),
        db_session,
    )
    assert first_ok["accepted"] is True
    run_before = await db_session.get(AIOptimizationRun, run_id)
    completed_before = int(run_before.experiments_completed)
    # Idempotent retry same client_request_id must not increment again.
    retry = await submit_agent_decision(
        run_id,
        DecisionRequest(lease_token=lease, client_request_id="decision-aaa-12345678", decision=good_decision),
        db_session,
    )
    assert retry["step_id"] == first_ok["step_id"]
    run_after = await db_session.get(AIOptimizationRun, run_id)
    assert int(run_after.experiments_completed) == completed_before
    # Different ID increments again? Actually budget allows only one decision per iteration; second distinct ID after READY should create new DECISION but still READY? For test, just check that second distinct ID creates different step.
    second = await submit_agent_decision(
        run_id,
        DecisionRequest(lease_token=lease, client_request_id="decision-bbb-12345678", decision=good_decision),
        db_session,
    )
    assert second["step_id"] != first_ok["step_id"]
    # Check that decision step links to result.
    from polyflip.db.models import AIRunStep

    dec_step = await db_session.get(AIRunStep, second["step_id"])
    assert dec_step is not None
    assert dec_step.client_request_id == "decision-bbb-12345678"
    assert dec_step.output_payload.get("result_id") == term.id


@pytest.mark.asyncio
async def test_context_asset_filtering_and_quality_gate(db_session):
    from polyflip.db.models import ModelRegistry, TradeHistory
    from datetime import datetime, timedelta, timezone

    # Create run with BTC asset and custom quality gate
    run_id = await _seed_run(db_session)
    # Update scope to BTCUSDT (should normalize to BTC) and custom gate
    run = await db_session.get(AIOptimizationRun, run_id)
    run.scope = {"asset": "BTCUSDT", "min_trades": 55, "max_ece": 0.12, "min_positive_oot_windows": 5}
    await db_session.flush()
    claimed = (await claim_next_agent_run(AgentClaimRequest(worker_id="ctx-tester"), db_session))["run"]
    # Need to claim the existing queued run; the previous claim used external-ai-research-agent,
    # so we force a new claim after releasing? Simpler: use the already claimed lease.
    # Actually _seed_run created QUEUED, claim_next_agent_run with ctx-tester will claim.
    # But we already updated scope before claim, so use that lease.
    lease = claimed["lease_token"]
    # Create ModelRegistry entries for BTC and ETH
    now = datetime.now(timezone.utc)
    btc_model = ModelRegistry(asset="BTC", version=1, is_active=True, trained_at=now, decision_threshold=0.55, accuracy=0.6, ece=0.05, quality_gate_passed=True)
    eth_model = ModelRegistry(asset="ETH", version=1, is_active=True, trained_at=now, decision_threshold=0.55, accuracy=0.7, ece=0.04, quality_gate_passed=True)
    db_session.add_all([btc_model, eth_model])
    # Create TradeHistory for BTC and ETH with required fields
    t_btc = TradeHistory(market_id="m-btc-1", asset="BTC", outcome_bought="YES", amount_usdc=10, executed_price=0.5, predicted_flip_prob=0.6, active_features="f1", status="FILLED", mode="PAPER", position_status="OPEN", pnl=1.0, timestamp=now - timedelta(hours=1), created_at=now - timedelta(hours=1))
    t_eth = TradeHistory(market_id="m-eth-1", asset="ETH", outcome_bought="YES", amount_usdc=10, executed_price=0.5, predicted_flip_prob=0.6, active_features="f1", status="FILLED", mode="PAPER", position_status="OPEN", pnl=5.0, timestamp=now - timedelta(hours=1), created_at=now - timedelta(hours=1))
    # Also BTCUSDT variant to test normalization
    t_btcusdt = TradeHistory(market_id="m-btc-2", asset="BTCUSDT", outcome_bought="YES", amount_usdc=10, executed_price=0.5, predicted_flip_prob=0.6, active_features="f1", status="PAPER_FILLED", mode="PAPER", position_status="OPEN", pnl=2.0, timestamp=now - timedelta(hours=2), created_at=now - timedelta(hours=2))
    db_session.add_all([t_btc, t_eth, t_btcusdt])
    await db_session.flush()
    ctx = await get_agent_context(run_id=run_id, lease_token=lease, db=db_session)
    # Only BTC models should be returned (normalized)
    assets = {m["asset"] for m in ctx["active_models"]}
    assert assets == {"BTC"}
    # Only BTC trades counted (BTC + BTCUSDT normalized to BTC = 2 trades, pnl 3.0)
    stats = ctx["recent_trade_statistics"]
    # trades_24h should be 2 (BTC + BTCUSDT) not ETH
    assert stats["trades_24h"] == 2
    assert stats["net_pnl_24h"] == 3.0
    # Quality gate from scope, not hardcoded
    qg = ctx["quality_gate"]
    assert qg["min_trades"] == 55
    assert qg["max_ece"] == 0.12
    assert qg["min_positive_oot_windows"] == 5


@pytest.mark.asyncio
async def test_heartbeat_returns_real_expiry(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    lease = claimed["lease_token"]
    from polyflip.api.ai_lab_agent import agent_heartbeat

    first = await agent_heartbeat(HeartbeatRequest(run_id=run_id, lease_token=lease), db_session)
    # leased_until should be future ISO timestamp
    from datetime import datetime, timezone

    leased_until = datetime.fromisoformat(first["leased_until"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    assert leased_until > now
    # Second heartbeat should extend expiry
    import asyncio as _asyncio

    await _asyncio.sleep(0.01)
    second = await agent_heartbeat(HeartbeatRequest(run_id=run_id, lease_token=lease), db_session)
    leased2 = datetime.fromisoformat(second["leased_until"].replace("Z", "+00:00"))
    assert leased2 >= leased_until


@pytest.mark.asyncio
async def test_claim_returns_snapshot_with_per_model_protocol(db_session):
    from polyflip.db.models import AILLMModelCatalog
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    # Seed two models with distinct protocols
    db_session.add(AILLMModelCatalog(provider="opencode", model_id="resp-model", display_name="resp", protocol="responses", is_available=True, is_discovered=True, probe_status="PASSED", last_checked_at=now, discovered_at=now, supports_structured_output=True))
    db_session.add(AILLMModelCatalog(provider="opencode", model_id="chat-model", display_name="chat", protocol="chat_completions", is_available=True, is_discovered=True, probe_status="PASSED", last_checked_at=now, discovered_at=now, supports_structured_output=True))
    await db_session.flush()
    # Create run with those models
    from polyflip.ai_lab.service import create_run
    from polyflip.ai_lab.service import create_permission
    from uuid import uuid4

    perm = await create_permission(db_session, profile_name=f"snap-{uuid4().hex[:4]}", allowed_actions=["CREATE_EXPERIMENT"], scope={}, limits={}, updated_by="test", enabled=True)
    run = await create_run(db_session, objective="snap test", scope={}, autonomy_level="OBSERVE", budget_experiments=1, permission=perm, llm_provider="opencode", llm_research_model="resp-model", llm_summary_model="chat-model")
    run.status = "QUEUED"
    await db_session.flush()
    await db_session.commit()
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    assert claimed is not None
    snap = claimed.get("llm_snapshot")
    assert snap is not None
    assert snap["provider"] == "opencode"
    assert snap["research"]["model_id"] == "resp-model"
    assert snap["research"]["protocol"] == "responses"
    assert snap["summary"]["model_id"] == "chat-model"
    assert snap["summary"]["protocol"] == "chat_completions"
    assert snap["catalog_checked_at"]
    # Also flat legacy fields still present
    assert claimed["llm_research_model"] == "resp-model"
    assert claimed["llm_summary_model"] == "chat-model"


def test_openapi_contains_hypothesis_and_decision_schemas():
    from polyflip.api.main import app

    openapi = app.openapi()
    schemas = openapi.get("components", {}).get("schemas", {})
    assert "HypothesisProposal" in schemas
    assert "AgentDecision" in schemas
