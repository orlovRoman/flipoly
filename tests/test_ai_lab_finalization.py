from types import SimpleNamespace

import pytest

from polyflip.ai_lab import orchestrator
from polyflip.ai_lab.service import AILabError


class _Session:
    def __init__(self, config=None, run=None):
        self.config = config
        self.run = run
        self.flush_count = 0

    async def get(self, model, object_id):
        name = getattr(model, "__name__", "")
        if name == "AIExperimentConfig":
            return self.config
        if name == "AIOptimizationRun":
            return self.run
        return None

    async def flush(self):
        self.flush_count += 1


def _ready_report(*, config_id=11, artifact_id=101):
    return {
        "recommendation_status": "READY_FOR_SHADOW",
        "recommended_config_id": config_id,
        "rows": [
            {
                "config_id": config_id,
                "artifact_ids": [artifact_id],
                "eligible_for_shadow": True,
            }
        ],
    }


@pytest.mark.asyncio
async def test_finalize_can_evaluate_without_shadow_assignment(monkeypatch):
    report = _ready_report()
    calls = []

    async def fake_evaluate(session, run_id):
        calls.append((session, run_id))
        return report

    async def unexpected_promote(*args, **kwargs):
        raise AssertionError("auto_shadow=False must not assign SHADOW")

    monkeypatch.setattr(orchestrator, "evaluate_run", fake_evaluate)
    monkeypatch.setattr(orchestrator, "promote_to_shadow", unexpected_promote)

    result = await orchestrator.finalize_run(
        object(),
        7,
        auto_shadow=False,
    )

    assert result == {"report": report, "assignment": None}
    assert calls == [(calls[0][0], 7)]


@pytest.mark.asyncio
async def test_finalize_assigns_reported_winner_to_shadow(monkeypatch):
    config = SimpleNamespace(asset="BTCUSDT", regime="low_vol")
    run = SimpleNamespace(summary=None)
    session = _Session(config=config, run=run)
    report = _ready_report(config_id=11, artifact_id=101)
    captured = {}

    async def fake_evaluate(db, run_id):
        assert db is session
        assert run_id == 7
        return report

    assignment = SimpleNamespace(
        id=42,
        candidate_artifact_id=101,
        baseline_artifact_id=None,
        asset="BTCUSDT",
        regime="low_vol",
    )

    async def fake_promote(db, **kwargs):
        assert db is session
        captured.update(kwargs)
        return assignment

    monkeypatch.setattr(orchestrator, "evaluate_run", fake_evaluate)
    monkeypatch.setattr(orchestrator, "promote_to_shadow", fake_promote)

    result = await orchestrator.finalize_run(session, 7)

    assert result["assignment"] is assignment
    assert captured == {
        "run_id": 7,
        "candidate_artifact_id": 101,
        "baseline_artifact_id": None,
        "asset": "BTCUSDT",
        "regime": "low_vol",
    }
    assert '"shadow_assignment"' in run.summary
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_finalize_rejects_shadow_when_asset_is_missing(monkeypatch):
    session = _Session(config=SimpleNamespace(asset=None, regime=None))
    report = _ready_report()

    async def fake_evaluate(db, run_id):
        return report

    async def unexpected_promote(*args, **kwargs):
        raise AssertionError("promotion must not happen without an asset")

    monkeypatch.setattr(orchestrator, "evaluate_run", fake_evaluate)
    monkeypatch.setattr(orchestrator, "promote_to_shadow", unexpected_promote)

    with pytest.raises(AILabError, match="asset is required"):
        await orchestrator.finalize_run(session, 7)
