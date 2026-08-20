import asyncio
from types import SimpleNamespace

import pytest

from polyflip.ai_lab import executor
from polyflip.ai_lab.service import AILabError


def test_registry_accepts_only_offline_actions():
    registry = executor.AdapterRegistry()
    adapter = lambda context: executor.AdapterResult(evaluation_kind="TRAIN")
    registry.register("TRAIN_MODEL", adapter)

    assert registry.get("TRAIN_MODEL") is adapter
    assert registry.actions() == ("TRAIN_MODEL",)

    with pytest.raises(AILabError):
        registry.register("ACTIVATE_MODEL", adapter)
    with pytest.raises(AILabError):
        registry.register("UNKNOWN_ACTION", adapter)
    with pytest.raises(AILabError):
        registry.register("TRAIN_MODEL", adapter)
    assert registry.unregister("TRAIN_MODEL") is adapter
    with pytest.raises(AILabError):
        registry.unregister("TRAIN_MODEL")


def test_adapter_result_rejects_wrong_evaluation_kind():
    result = executor.AdapterResult(evaluation_kind="OOT")
    with pytest.raises(AILabError):
        result.validate_for("TRAIN_MODEL")


class _FakeSession:
    def __init__(self, run, config):
        self.run = run
        self.config = config
        self.commits = 0
        self.rollbacks = 0
        self.added = []

    def add(self, value):
        self.added.append(value)

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
    assert outcome.config_id == 7
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


def test_transient_adapter_failure_returns_retry_wait_and_reopens_step(monkeypatch):
    step = SimpleNamespace(
        id=45,
        action="TRAIN_MODEL",
        input_payload={"config_id": 10},
        status="RUNNING",
        finished_at=None,
        summary=None,
        error_code=None,
        error_message=None,
    )
    run = SimpleNamespace(objective="retry transient failure", scope={})
    config = SimpleNamespace(config_hash="hash-10")
    session = _FakeSession(run, config)
    job = SimpleNamespace(id=77, status="QUEUED", owner_token="worker-a")
    completed = {}

    async def claim(_session, _run_id):
        return step

    async def ensure(_session, **_kwargs):
        return job

    async def claim_job(_session, _key, **_kwargs):
        job.status = "RUNNING"
        return job

    async def record(_session, **kwargs):
        return SimpleNamespace(id=101, **{key: kwargs.get(key) for key in ()})

    async def complete(_session, _job_id, **kwargs):
        completed.update(kwargs)
        job.status = kwargs["status"]
        return job

    async def adapter(_context):
        raise RuntimeError("HTTP 503 service unavailable")

    monkeypatch.setattr(executor, "claim_next_step", claim)
    monkeypatch.setattr(executor, "ensure_job", ensure)
    monkeypatch.setattr(executor, "claim_job", claim_job)
    monkeypatch.setattr(executor, "record_result", record)
    monkeypatch.setattr(executor, "complete_job", complete)

    registry = executor.AdapterRegistry().register("TRAIN_MODEL", adapter)
    outcome = asyncio.run(
        executor.execute_next_step(
            session,
            7,
            registry,
            owner_token="worker-a",
        )
    )

    assert outcome is not None
    assert outcome.status == "FAILED"
    assert completed["status"] == "RETRY_WAIT"
    assert job.status == "RETRY_WAIT"
    assert step.status == "PENDING"


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
    assert outcome.config_id == 9
    assert step.error_code == "INVALID_STEP_INPUT"
    assert session.added[0].error_code == "INVALID_STEP_INPUT"


def test_malformed_config_id_is_audited_without_zero_sentinel(monkeypatch):
    step = SimpleNamespace(
        id=44,
        action="TRAIN_MODEL",
        input_payload={"config_id": "not-an-id"},
        status="RUNNING",
        finished_at=None,
        summary=None,
        error_code=None,
        error_message=None,
    )
    run = SimpleNamespace(objective="offline only", scope={})
    session = _FakeSession(run, None)

    async def claim(_session, _run_id):
        return step

    monkeypatch.setattr(executor, "claim_next_step", claim)
    outcome = asyncio.run(
        executor.execute_next_step(session, 6, executor.AdapterRegistry())
    )

    assert outcome is not None
    assert outcome.status == "FAILED"
    assert outcome.config_id is None
    assert session.added[0].config_id is None


def test_execute_steps_preserves_completed_outcomes_on_error(monkeypatch):
    completed = SimpleNamespace(status="SUCCEEDED")
    calls = 0

    async def run_one(_session, _run_id, _registry):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completed
        raise RuntimeError("persist failed")

    monkeypatch.setattr(executor, "execute_next_step", run_one)

    with pytest.raises(executor.ExecutionBatchError) as raised:
        asyncio.run(
            executor.execute_steps(
                object(),
                1,
                executor.AdapterRegistry(),
                max_steps=2,
            )
        )

    assert raised.value.completed == (completed,)
    assert isinstance(raised.value.cause, RuntimeError)
