"""Regression coverage for legacy AI Lab runs with a null iteration counter."""

from types import SimpleNamespace

from polyflip.api.ai_lab_agent import _claimed_run_payload
from services.ai_research_agent.schemas import ClaimedRun


def test_claimed_run_normalizes_null_experiments_completed() -> None:
    payload = {
        "id": 5,
        "status": "RUNNING",
        "experiments_completed": None,
    }

    claimed = ClaimedRun.model_validate(payload)

    assert claimed.experiments_completed == 0


def test_claim_payload_normalizes_null_experiments_completed() -> None:
    run = SimpleNamespace(
        id=5,
        status="RUNNING",
        objective="Test",
        scope={},
        mode="RESEARCH",
        autonomy_level="EXPERIMENT",
        budget_experiments=1,
        budget_seconds=0,
        experiments_completed=None,
        llm_provider="opencode",
        llm_research_model="gpt-5.6-luna",
        llm_summary_model="big-pickle",
        llm_snapshot={},
    )

    payload = _claimed_run_payload(run, "lease-token")

    assert payload["experiments_completed"] == 0
