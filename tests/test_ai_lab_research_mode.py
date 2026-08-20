from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polyflip.ai_lab.orchestrator import build_experiment_report
import polyflip.ai_lab.service as ai_service
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
    assert row["candidate_status"] == "PROVISIONAL"
    assert row["evidence_status"] == "INSUFFICIENT"
    assert row["deployment_status"] == "PROHIBITED"
    assert report["recommendation_status"] == "RESEARCH_PROVISIONAL"
    assert report["candidate_status"] == "PROVISIONAL"


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


@pytest.mark.asyncio
async def test_research_allowed_in_paper_but_blocked_by_live_gate(monkeypatch):
    session = SimpleNamespace(add=lambda row: None, flush=AsyncMock())
    permission = SimpleNamespace(id=8, enabled=True)
    monkeypatch.setattr(ai_service.settings, "TRADING_ENABLED", True)
    monkeypatch.setattr(ai_service.settings, "LIVE_TRADING_ENABLED", False)
    run = await create_run(
        session,
        objective="paper research",
        scope={"asset": "BTC"},
        autonomy_level="EXPERIMENT",
        budget_experiments=1,
        permission=permission,
        mode="RESEARCH",
    )
    assert run.mode == "RESEARCH"

    monkeypatch.setattr(ai_service.settings, "LIVE_TRADING_ENABLED", True)
    with pytest.raises(Exception, match="LIVE_TRADING_ENABLED"):
        await create_run(
            session,
            objective="live blocked research",
            scope={"asset": "BTC"},
            autonomy_level="EXPERIMENT",
            budget_experiments=1,
            permission=permission,
            mode="RESEARCH",
        )
