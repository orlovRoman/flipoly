from types import SimpleNamespace

import pytest

from polyflip.ai_lab import orchestrator
from polyflip.ai_lab.service import AILabError


class _Session:
    def __init__(self, config=None, run=None, step=None):
        self.config = config
        self.run = run
        self.step = step
        self.added = []
        self.flush_count = 0

    async def get(self, model, object_id):
        name = getattr(model, "__name__", "")
        if name == "AIExperimentConfig":
            return self.config
        if name == "AIOptimizationRun":
            return self.run
        if name == "AIRunStep":
            return self.step
        return None

    def add(self, item):
        self.added.append(item)

    async def execute(self, stmt):
        class _Result:
            def __init__(self, item):
                self._item = item

            def scalars(self):
                return self

            def all(self):
                return [self._item] if self._item is not None else []

            def scalar_one_or_none(self):
                return self._item

        return _Result(self.step)

    async def flush(self):
        self.flush_count += 1


def _ready_report(*, config_id=11, artifact_id=101):
    return {
        "recommendation_status": "READY_FOR_SHADOW",
        "rejection_reasons": [],
        "recommended_config_id": config_id,
        "window_count": 3,
        "total_trades": 74,
        "median_pnl": 1.24,
        "rows": [
            {
                "config_id": config_id,
                "artifact_ids": [artifact_id],
                "eligible_for_shadow": True,
                "window_count": 3,
                "total_trades": 74,
                "median_oot_pnl": 1.24,
            }
        ],
    }


def _rejected_report(*, status="INSUFFICIENT_TRADES", reasons=None):
    reasons = reasons or [status]
    return {
        "recommendation_status": status,
        "rejection_reasons": reasons,
        "recommended_config_id": None,
        "window_count": 3,
        "total_trades": 40,
        "median_pnl": 1.24,
        "rows": [
            {
                "config_id": 11,
                "artifact_ids": [101],
                "eligible_for_shadow": False,
                "rejection_reasons": reasons,
                "window_count": 3,
                "total_trades": 40,
                "median_oot_pnl": 1.24,
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
        _Session(run=SimpleNamespace(summary=None)),
        7,
        auto_shadow=False,
    )

    assert result == {"report": report, "assignment": None}
    assert len(calls) == 1
    assert calls[0][1] == 7


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


@pytest.mark.asyncio
async def test_finalize_records_audit_log_and_summary_on_gate_rejection(monkeypatch):
    run = SimpleNamespace(summary=None)
    step = SimpleNamespace(id=10, step_index=2)
    session = _Session(run=run, step=step)
    report = _rejected_report(status="INSUFFICIENT_TRADES")

    async def fake_evaluate(db, run_id):
        return report

    async def unexpected_promote(*args, **kwargs):
        raise AssertionError("rejected run must not call promote_to_shadow")

    monkeypatch.setattr(orchestrator, "evaluate_run", fake_evaluate)
    monkeypatch.setattr(orchestrator, "promote_to_shadow", unexpected_promote)

    result = await orchestrator.finalize_run(session, 7)

    assert result["assignment"] is None
    assert result["report"]["recommendation_status"] == "INSUFFICIENT_TRADES"
    assert '"INSUFFICIENT_TRADES"' in run.summary
    assert len(session.added) == 1
    audit_entry = session.added[0]
    assert audit_entry.action == "FINALIZE_RUN"
    assert audit_entry.error_code == "GATE_REJECTED_INSUFFICIENT_TRADES"
