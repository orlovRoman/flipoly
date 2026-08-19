from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polyflip.ai_lab.orchestrator import build_experiment_report
from polyflip.ai_lab.policy import evaluate_research_policy, validate_artifact_technical
from polyflip.ai_lab.service import create_run


@pytest.mark.parametrize(
    ("metrics", "status"),
    [
        ({"median_pnl": 1.0, "total_trades": 1, "window_count": 1}, "INSUFFICIENT_EVIDENCE"),
        ({"median_pnl": -1.0, "total_trades": 50, "window_count": 3}, "RESEARCH_PROVISIONAL"),
    ],
)
def test_research_policy_retains_weak_candidates(metrics, status):
    result = evaluate_research_policy(metrics)

    assert result.technical_valid is True
    assert result.deployment_eligible is False
    assert result.recommendation_status == status


def test_research_policy_rejects_non_finite_metrics_as_technical_invalid():
    result = evaluate_research_policy({"median_pnl": float("nan"), "trade_count": 1})

    assert result.technical_valid is False
    assert result.recommendation_status == "TECHNICAL_INVALID"


def test_malformed_artifact_is_rejected():
    valid, reasons = validate_artifact_technical(
        SimpleNamespace(loadability_status="INVALID", artifact_metadata=None)
    )

    assert valid is False
    assert "INVALID_ARTIFACT_LOADABILITY" in reasons
    assert "MALFORMED_ARTIFACT_METADATA" in reasons


def test_research_report_keeps_negative_pnl_and_separates_evidence():
    results = [
        {
            "config_id": 1,
            "evaluation_kind": "POLYMARKET_OOT",
            "status": "SUCCEEDED",
            "net_pnl": -2.0,
            "trade_count": 1,
            "max_drawdown": 0.5,
            "oot_window_start": "2026-01-01T00:00:00+00:00",
            "oot_window_end": "2026-01-02T00:00:00+00:00",
            "metrics": {},
        }
    ]

    report = build_experiment_report(results, mode="RESEARCH")
    row = report["rows"][0]

    assert report["mode"] == "RESEARCH"
    assert row["median_oot_pnl"] == -2.0
    assert row["technical_valid"] is True
    assert row["evidence_sufficient"] is False
    assert row["deployment_eligible"] is False
    assert report["recommendation_status"] == "RESEARCH_PROVISIONAL"


@pytest.mark.asyncio
async def test_research_run_is_explicitly_marked_and_does_not_activate():
    session = SimpleNamespace(add=lambda row: None, flush=AsyncMock())
    permission = SimpleNamespace(id=7, enabled=True)

    run = await create_run(
        session,
        objective="research candidate",
        scope={"asset": "BTC"},
        autonomy_level="EXPERIMENT",
        budget_experiments=1,
        permission=permission,
        mode="RESEARCH",
    )

    assert run.mode == "RESEARCH"
    assert run.status == "DRAFT"
