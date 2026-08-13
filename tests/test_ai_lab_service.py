from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polyflip.ai_lab.service import (
    AILabError,
    AIPermissionError,
    AIRunTransitionError,
    LAB_ACTIONS,
    create_run,
    transition_action_for_target,
    transition_run,
    validate_permission,
    validate_run_transition,
)


def test_run_transition_graph_requires_evaluation_before_shadow():
    validate_run_transition("DRAFT", "PLANNING")
    validate_run_transition("PLANNING", "RUNNING")
    validate_run_transition("RUNNING", "EVALUATING")
    validate_run_transition("EVALUATING", "SHADOW")
    with pytest.raises(AIRunTransitionError):
        validate_run_transition("DRAFT", "SHADOW")


def test_terminal_runs_cannot_be_restarted():
    with pytest.raises(AIRunTransitionError):
        validate_run_transition("FAILED", "RUNNING")


def test_permission_is_allow_list_and_disabled_profiles_are_rejected():
    permission = SimpleNamespace(
        profile_name="experiment-only",
        version=1,
        enabled=True,
        allowed_actions=["CREATE_EXPERIMENT", "RUN_OOT_BACKTEST"],
    )
    validate_permission(permission, "CREATE_EXPERIMENT")
    with pytest.raises(AIPermissionError):
        validate_permission(permission, "REQUEST_ACTIVATION")
    permission.enabled = False
    with pytest.raises(AIPermissionError):
        validate_permission(permission, "CREATE_EXPERIMENT")


@pytest.mark.asyncio
async def test_create_run_persists_permission_snapshot_without_live_side_effects():
    session = SimpleNamespace(add=lambda row: None, flush=AsyncMock())
    permission = SimpleNamespace(id=7, profile_name="experiment-only", version=1, enabled=True,
                                 allowed_actions=["CREATE_EXPERIMENT"])
    run = await create_run(
        session,
        objective="compare A/B/C",
        scope={"asset": "BTCUSDT"},
        autonomy_level="EXPERIMENT",
        budget_experiments=3,
        permission=permission,
    )
    assert run.status == "DRAFT"
    assert run.permission_id == 7
    assert run.autonomy_level == "EXPERIMENT"
    session.flush.assert_awaited_once()


def test_lab_action_set_does_not_include_live_activation():
    assert "ACTIVATE_LIVE" not in LAB_ACTIONS


def test_public_transition_actions_are_allow_listed():
    assert transition_action_for_target("PLANNING") == "CREATE_EXPERIMENT"
    assert transition_action_for_target("SHADOW") == "PROMOTE_TO_SHADOW"
    assert transition_action_for_target("ACTIVE") is None


@pytest.mark.asyncio
async def test_active_transition_is_blocked_without_human_approval():
    run = SimpleNamespace(status="PENDING_APPROVAL")
    session = SimpleNamespace(flush=AsyncMock())
    with pytest.raises(AIRunTransitionError, match="human approval"):
        await transition_run(session, run, "ACTIVE")
    session.flush.assert_not_awaited()
