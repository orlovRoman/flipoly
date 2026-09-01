"""Lease recovery, status and terminal-result tests for the external agent API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

import pytest
import sqlalchemy as sa
from fastapi import HTTPException

from polyflip.ai_lab.service import create_permission, create_run
from polyflip.api.ai_lab import get_ai_agent_status
from polyflip.api.ai_lab_agent import (
    AgentClaimRequest,
    CompleteRequest,
    DecisionRequest,
    ProposalRequest,
    claim_next_agent_run,
    complete_agent_run,
    get_agent_phase,
    submit_agent_decision,
    submit_agent_proposal,
)
from polyflip.db.models import AIConfigOverlay, AIModelArtifact, AIOptimizationRun, AIRunStep, AIWorkerLease, ExperimentResult


def _proposal() -> dict:
    return {
        "hypothesis": "Use a calibrated outsider model to improve OOT PnL",
        "asset": "BTC",
        "market_role": "OUTSIDER",
        "model_family": "LOGREG",
        "feature_set": "FS_D1",
        "parameter_changes": {"C": 0.5},
        "strategy_parameter_changes": {"decision_threshold": 0.58},
        "expected_effect": {
            "metric": "median_oot_pnl",
            "direction": "increase",
            "target_gain": 0.05,
        },
        "reasoning": ["baseline drift"],
        "risks": ["small sample"],
        "test_plan": {
            "oot_windows": 3,
            "min_markets": 50,
            "execution_mode": "PAPER_REALISTIC",
        },
    }


async def _seed_run(
    db_session,
    *,
    actions: list[str] | None = None,
    autonomy: str = "EXPERIMENT",
) -> int:
    permission = await create_permission(
        db_session,
        profile_name="recovery-test-" + uuid.uuid4().hex[:8],
        allowed_actions=actions or ["CREATE_EXPERIMENT", "TRAIN_MODEL"],
        scope={},
        limits={},
        updated_by="test",
        enabled=True,
    )
    run = await create_run(
        db_session,
        objective="external agent recovery",
        scope={"asset": "BTC"},
        autonomy_level=autonomy,
        budget_experiments=2,
        permission=permission,
        llm_provider="mock",
    )
    run.status = "QUEUED"
    await db_session.flush()
    await db_session.commit()
    return int(run.id)


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed_and_old_token_is_rejected(db_session):
    run_id = await _seed_run(db_session)
    first = (await claim_next_agent_run(AgentClaimRequest(worker_id="worker-a"), db_session))["run"]
    old_token = first["lease_token"]
    lease = (await db_session.execute(sa.select(AIWorkerLease))).scalar_one()
    lease.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await db_session.commit()

    second = (await claim_next_agent_run(AgentClaimRequest(worker_id="worker-b"), db_session))["run"]
    assert second["id"] == run_id
    assert second["lease_token"] != old_token

    with pytest.raises(HTTPException) as exc_info:
        await get_agent_phase(run_id, lease_token=old_token, db=db_session)
    assert exc_info.value.detail == "LEASE_LOST"
    phase = await get_agent_phase(
        run_id, lease_token=second["lease_token"], db=db_session
    )
    assert phase["phase"] == "NEEDS_PROPOSAL"


@pytest.mark.asyncio
async def test_status_endpoint_exposes_queue_and_active_overlays(db_session):
    run_id = await _seed_run(db_session)
    now = datetime.now(timezone.utc)
    overlay = AIConfigOverlay(
        run_id=run_id,
        scope={"target": "PAPER", "asset": "BTC"},
        changes={"DEAD_ZONE_WIDTH": 0.06},
        base_settings_hash="a" * 64,
        resulting_settings_hash="b" * 64,
        status="APPLIED",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=5),
    )
    db_session.add(overlay)
    await db_session.flush()

    payload = await get_ai_agent_status(db_session)

    assert payload["state"] == "idle"
    assert payload["queue_count"] == 1
    assert [item["id"] for item in payload["active_overlays"]] == [overlay.id]
    assert payload["active_overlays"][0]["metrics"]["after"]["coverage"] == 0.0


@pytest.mark.asyncio
async def test_apply_overlay_decision_persists_effect_and_scope(db_session):
    run_id = await _seed_run(
        db_session,
        actions=["CREATE_EXPERIMENT", "TRAIN_MODEL", "APPLY_CONFIG_OVERLAY"],
        autonomy="AUTONOMOUS_CONFIG",
    )
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    proposal_result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=_proposal(),
        ),
        db_session,
    )
    result = ExperimentResult(
        run_id=run_id,
        config_id=proposal_result["config_id"],
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        net_pnl=1.5,
        trade_count=100,
        max_drawdown=2.0,
        oot_window_start=datetime.now(timezone.utc) - timedelta(days=1),
        oot_window_end=datetime.now(timezone.utc),
    )
    db_session.add(result)
    await db_session.flush()
    request_id = uuid.uuid4().hex

    response = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=claimed["lease_token"],
            client_request_id=request_id,
            decision={
                "action": "APPLY_OVERLAY",
                "rationale": "The candidate improves the measured result",
                "key_findings": ["positive OOT PnL"],
                "recommended_config_id": proposal_result["config_id"],
                "proposed_overlay": {"DEAD_ZONE_WIDTH": 0.06},
                "next_step_focus": None,
            },
        ),
        db_session,
    )

    assert response["accepted"] is True
    overlay_id = response["effects"]["overlay_id"]
    overlay = await db_session.get(AIConfigOverlay, overlay_id)
    assert overlay.status == "APPLIED"
    assert overlay.scope == {"target": "SHADOW_SIMULATION", "asset": "BTC"}
    decision_step = (
        await db_session.execute(
            sa.select(AIRunStep).where(
                AIRunStep.run_id == run_id, AIRunStep.step_type == "DECISION"
            )
        )
    ).scalar_one()
    assert decision_step.output_payload["effects"]["overlay_id"] == overlay_id


@pytest.mark.asyncio
async def test_negative_oot_can_be_finalized_without_shadow_promotion(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    proposal_result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=_proposal(),
        ),
        db_session,
    )
    db_session.add(
        ExperimentResult(
            run_id=run_id,
            config_id=proposal_result["config_id"],
            evaluation_kind="POLYMARKET_OOT",
            status="SUCCEEDED",
            net_pnl=-2.0,
            trade_count=100,
            max_drawdown=8.0,
            oot_window_start=datetime.now(timezone.utc) - timedelta(days=1),
            oot_window_end=datetime.now(timezone.utc),
        )
    )
    await db_session.flush()

    decision = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            decision={
                "action": "FINALIZE_NO_WINNER",
                "rationale": "Negative OOT PnL rejects the candidate",
                "key_findings": ["net PnL is negative"],
                "recommended_config_id": None,
                "proposed_overlay": None,
                "next_step_focus": None,
            },
        ),
        db_session,
    )
    assert decision["accepted"] is True
    completed = await complete_agent_run(
        run_id,
        CompleteRequest(
            action="COMPLETED",
            reason="candidate rejected",
            lease_token=claimed["lease_token"],
        ),
        db_session,
    )
    assert completed["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_failed_oot_is_decidable_after_restart(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    proposal_result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=_proposal(),
        ),
        db_session,
    )
    db_session.add(
        ExperimentResult(
            run_id=run_id,
            config_id=proposal_result["config_id"],
            evaluation_kind="POLYMARKET_OOT",
            status="FAILED",
            error_code="NO_DATA",
            error_message="provider returned no markets",
        )
    )
    pipeline_steps = (
        (await db_session.execute(sa.select(AIRunStep).where(AIRunStep.run_id == run_id)))
        .scalars()
        .all()
    )
    for step in pipeline_steps:
        if step.step_type in {"TRAIN_MODEL", "RUN_OOT_BACKTEST", "RUN_POLYMARKET_OOT"}:
            step.status = "FAILED"
    await db_session.flush()

    phase = await get_agent_phase(
        run_id, lease_token=claimed["lease_token"], db=db_session
    )
    assert phase["phase"] == "NEEDS_DECISION"
    assert phase["latest_result_id"] is not None
    decision = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            decision={
                "action": "FINALIZE_NO_WINNER",
                "rationale": "The failed evaluation cannot support promotion",
                "key_findings": ["evaluation failed"],
                "recommended_config_id": None,
                "proposed_overlay": None,
                "next_step_focus": None,
            },
        ),
        db_session,
    )
    assert decision["accepted"] is True

@pytest.mark.asyncio
async def test_recommend_shadow_canonicalizes_running_status(db_session, monkeypatch):
    run_id = await _seed_run(
        db_session,
        actions=["CREATE_EXPERIMENT", "TRAIN_MODEL", "PROMOTE_TO_SHADOW"],
        autonomy="AUTONOMOUS_SHADOW",
    )
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    proposal_result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=_proposal(),
        ),
        db_session,
    )
    db_session.add(
        ExperimentResult(
            run_id=run_id,
            config_id=proposal_result["config_id"],
            evaluation_kind="POLYMARKET_OOT",
            status="SUCCEEDED",
            net_pnl=1.0,
            trade_count=100,
            max_drawdown=1.0,
            oot_window_start=datetime.now(timezone.utc) - timedelta(days=1),
            oot_window_end=datetime.now(timezone.utc),
        )
    )
    await db_session.flush()

    seen_statuses: list[str] = []

    async def fake_finalize(db, run_id, **kwargs):
        current = await db.get(AIOptimizationRun, run_id)
        seen_statuses.append(current.status)
        return {
            "assignment": None,
            "report": {"shadow_recommendation_status": "RESEARCH_PROVISIONAL"},
        }

    monkeypatch.setattr("polyflip.api.ai_lab_agent.finalize_run", fake_finalize)
    response = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            decision={
                "action": "RECOMMEND_SHADOW",
                "rationale": "The candidate is ready for passive evaluation",
                "key_findings": ["positive OOT PnL"],
                "recommended_config_id": proposal_result["config_id"],
                "proposed_overlay": None,
                "next_step_focus": None,
            },
        ),
        db_session,
    )
    assert response["accepted"] is True
    assert seen_statuses == ["EVALUATING"]

@pytest.mark.asyncio
async def test_recommend_shadow_promotes_passed_research_candidate(db_session):
    from polyflip.db.models import AIShadowAssignment

    run_id = await _seed_run(
        db_session,
        actions=[
            "CREATE_EXPERIMENT",
            "TRAIN_MODEL",
            "RUN_OOT_BACKTEST",
            "PROMOTE_TO_SHADOW",
        ],
        autonomy="AUTONOMOUS_SHADOW",
    )
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    proposal_result = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            proposal=_proposal(),
        ),
        db_session,
    )
    config_id = proposal_result["config_id"]
    artifact = AIModelArtifact(
        config_id=config_id,
        run_id=run_id,
        artifact_hash="a" * 64,
        sha256="b" * 64,
        schema_version="1",
        feature_pipeline_version="agent-v1",
        artifact_metadata={"config_id": config_id},
        loadability_status="VALID",
    )
    db_session.add(artifact)
    await db_session.flush()
    now = datetime.now(timezone.utc)
    for offset in range(3):
        db_session.add(
            ExperimentResult(
                run_id=run_id,
                config_id=config_id,
                artifact_id=artifact.id,
                evaluation_kind="POLYMARKET_OOT",
                status="SUCCEEDED",
                net_pnl=1.0 + offset,
                trade_count=20,
                max_drawdown=1.0,
                oot_window_start=now - timedelta(days=offset + 2),
                oot_window_end=now - timedelta(days=offset + 1),
            )
        )
    await db_session.flush()

    response = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=claimed["lease_token"],
            client_request_id=uuid.uuid4().hex,
            decision={
                "action": "RECOMMEND_SHADOW",
                "rationale": "The candidate passed all research gate checks",
                "key_findings": ["positive PnL across three windows"],
                "recommended_config_id": config_id,
                "proposed_overlay": None,
                "next_step_focus": None,
            },
        ),
        db_session,
    )

    assert response["accepted"] is True
    assert response["effects"]["shadow_assignment_id"]
    stored_run = await db_session.get(AIOptimizationRun, run_id)
    assert stored_run.status == "SHADOW"
    assignment = (
        await db_session.execute(
            sa.select(AIShadowAssignment).where(AIShadowAssignment.run_id == run_id)
        )
    ).scalar_one()
    assert assignment.candidate_artifact_id == artifact.id
