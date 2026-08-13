import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polyflip.ai_lab import executor


def test_registry_accepts_only_offline_actions():
    registry = executor.AdapterRegistry()
    adapter = lambda context: executor.AdapterResult(evaluation_kind="TRAIN")
    registry.register("TRAIN_MODEL", adapter)

    assert registry.get("TRAIN_MODEL") is adapter
    assert registry.actions() == ("TRAIN_MODEL",)

    with pytest.raises(Exception):
        registry.register("ACTIVATE_MODEL", adapter)
    with pytest.raises(Exception):
        registry.register("UNKNOWN_ACTION", adapter)


def test_adapter_result_rejects_wrong_evaluation_kind():
    result = executor.AdapterResult(evaluation_kind="OOT")
    with pytest.raises(Exception):
        result.validate_for("TRAIN_MODEL")


class _FakeSession:
    def __init__(self, run, config):
        self.run = run
        self.config = config
        self.commits = 0
        self.rollbacks = 0

    async def get(self, model, identifier):
        if model is executor.AIOptimizationRun:
            return self.run
        if model is executor.AIExperimentConfig:
            return self.config
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def test_missing_adapter_is_recorded_as_failed_result(monkeypatch):
    step = SimpleNamespace(
        id=41,
        action="RUN_OOT_BACKTEST",
        input_payload={"config_id": 7},
    )
    run = SimpleNamespace(objective="compare variants", scope={})
    config = SimpleNamespace(config_hash="hash-7")
    session = _FakeSession(run, config)
    recorded = {}

    async def claim(_session, _run_id):
        return step

    async def record(_session, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(id=99)

    monkeypatch.setattr(executor, "claim_next_step", claim)
    monkeypatch.setattr(executor, "record_result", record)

    outcome = asyncio.run(
        executor.execute_next_step(
            session,
            3,
            executor.AdapterRegistry(),
        )
    )

    assert outcome is not None
    assert outcome.status == "FAILED"
    assert outcome.result_id == 99
    assert outcome.error_code == "ADAPTER_NOT_REGISTERED"
    assert recorded["evaluation_kind"] == "OOT"
    assert recorded["status"] == "FAILED"
    assert recorded["step_id"] == 41
    assert recorded["error_code"] == "ADAPTER_NOT_REGISTERED"
    assert session.commits == 2


def test_adapter_runs_after_claim_commit_and_persists_result(monkeypatch):
    step = SimpleNamespace(
        id=42,
        action="RUN_POLYMARKET_OOT",
        input_payload={"config_id": 8},
    )
    run = SimpleNamespace(objective="polymarket test", scope={"min_trades": 3})
    config = SimpleNamespace(config_hash="hash-8")
    session = _FakeSession(run, config)
    events = []
    recorded = {}

    async def claim(_session, _run_id):
        return step

    async def adapter(context):
        events.append(("adapter", session.commits, context.config_id))
        return executor.AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            metrics={"auc": 0.74},
            trade_count=12,
            net_pnl=2.5,
            max_drawdown=-0.8,
            summary="real-price OOT completed",
        )

    async def record(_session, **kwargs):
        events.append(("record", session.commits))
        recorded.update(kwargs)
        return SimpleNamespace(id=100)

    registry = executor.AdapterRegistry().register("RUN_POLYMARKET_OOT", adapter)
    monkeypatch.setattr(executor, "claim_next_step", claim)
    monkeypatch.setattr(executor, "record_result", record)

    outcome = asyncio.run(executor.execute_next_step(session, 4, registry))

    assert outcome is not None
    assert outcome.status == "SUCCEEDED"
    assert outcome.result_id == 100
    assert events[0] == ("adapter", 1, 8)
    assert events[1] == ("record", 1)
    assert recorded["trade_count"] == 12
    assert recorded["net_pnl"] == 2.5
    assert recorded["summary"] == "real-price OOT completed"
    assert session.commits == 2


def test_unknown_claimed_action_is_closed_without_key_error(monkeypatch):
    step = SimpleNamespace(
        id=43,
        action="PLACE_ORDER",
        input_payload={"config_id": 9},
        status="RUNNING",
        finished_at=None,
        summary=None,
        error_code=None,
        error_message=None,
    )
    run = SimpleNamespace(objective="offline only", scope={})
    config = SimpleNamespace(config_hash="hash-9")
    session = _FakeSession(run, config)

    async def claim(_session, _run_id):
        return step

    monkeypatch.setattr(executor, "claim_next_step", claim)

    outcome = asyncio.run(
        executor.execute_next_step(session, 5, executor.AdapterRegistry())
    )

    assert outcome is not None
    assert outcome.status == "FAILED"
    assert outcome.error_code == "INVALID_STEP_INPUT"
    assert step.error_code == "INVALID_STEP_INPUT"
