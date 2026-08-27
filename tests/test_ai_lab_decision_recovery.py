import uuid
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

from polyflip.ai_lab.service import create_permission, create_run
from polyflip.api.ai_lab_agent import (
    AgentClaimRequest,
    DecisionRequest,
    ProposalRequest,
    claim_next_agent_run,
    submit_agent_decision,
    submit_agent_proposal,
)
from polyflip.db.models import AIOptimizationRun, AIRunStep, ExperimentResult


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


async def _seed_run(db_session, *, actions=None, autonomy="OBSERVE") -> int:
    permission = await create_permission(
        db_session,
        profile_name="decision-recovery-" + uuid.uuid4().hex[:8],
        allowed_actions=actions or ["CREATE_EXPERIMENT", "TRAIN_MODEL"],
        scope={},
        limits={},
        updated_by="test",
        enabled=True,
    )
    run = await create_run(
        db_session,
        objective="decision recovery",
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


async def _prepare_result(db_session, run_id: int, lease: str) -> int:
    proposal = await submit_agent_proposal(
        run_id,
        ProposalRequest(
            lease_token=lease,
            client_request_id=uuid.uuid4().hex,
            proposal=_proposal(),
        ),
        db_session,
    )
    db_session.add(
        ExperimentResult(
            run_id=run_id,
            config_id=proposal["config_id"],
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
    return proposal["config_id"]


@pytest.mark.asyncio
async def test_unknown_recommended_config_is_ignored_with_warning(db_session):
    run_id = await _seed_run(db_session)
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    lease = claimed["lease_token"]
    await _prepare_result(db_session, run_id, lease)

    response = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=lease,
            client_request_id=uuid.uuid4().hex,
            decision={
                "action": "CONTINUE_RESEARCH",
                "rationale": "Keep collecting evidence",
                "key_findings": ["result is promising"],
                "recommended_config_id": 999999999,
                "proposed_overlay": None,
                "next_step_focus": "more windows",
            },
        ),
        db_session,
    )

    assert response["accepted"] is True
    assert response["effects"]["warnings"] == ["unknown recommended_config_id ignored"]
    step = await db_session.get(AIRunStep, response["step_id"])
    assert step.output_payload["recommended_config_id"] is None


@pytest.mark.asyncio
async def test_experiment_recommend_shadow_is_recorded_for_manual_assignment(
    db_session,
):
    run_id = await _seed_run(
        db_session,
        actions=["CREATE_EXPERIMENT", "TRAIN_MODEL", "PROMOTE_TO_SHADOW"],
        autonomy="EXPERIMENT",
    )
    claimed = (await claim_next_agent_run(AgentClaimRequest(), db_session))["run"]
    lease = claimed["lease_token"]
    config_id = await _prepare_result(db_session, run_id, lease)

    response = await submit_agent_decision(
        run_id,
        DecisionRequest(
            lease_token=lease,
            client_request_id=uuid.uuid4().hex,
            decision={
                "action": "RECOMMEND_SHADOW",
                "rationale": "The candidate merits review",
                "key_findings": ["positive OOT result"],
                "recommended_config_id": config_id,
                "proposed_overlay": None,
                "next_step_focus": None,
            },
        ),
        db_session,
    )

    assert response["accepted"] is True
    assert response["effects"] == {
        "shadow_status": "PENDING_MANUAL_ASSIGNMENT",
        "autonomy_blocked": True,
        "reason": "autonomy level does not permit automatic shadow assignment",
    }
    stored_run = await db_session.get(AIOptimizationRun, run_id)
    assert stored_run.status == "EVALUATING"
    assert (
        await db_session.execute(
            sa.select(AIRunStep).where(
                AIRunStep.run_id == run_id, AIRunStep.step_type == "DECISION"
            )
        )
    ).scalar_one().output_payload["effects"]["autonomy_blocked"] is True
