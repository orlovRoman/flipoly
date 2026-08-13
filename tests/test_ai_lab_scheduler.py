import asyncio
from types import SimpleNamespace

import pytest

from polyflip.ai_lab import scheduler
from polyflip.ai_lab.executor import ExecutionOutcome


class _Session:
    async def get(self, _model, _run_id):
        return SimpleNamespace(status="RUNNING")


def _outcome(status="SUCCEEDED"):
    return ExecutionOutcome(
        run_id=1,
        step_id=10,
        action="TRAIN_MODEL",
        evaluation_kind="TRAIN",
        status=status,
        result_id=11,
        config_id=3,
    )


def test_scheduler_limits_are_bounded():
    with pytest.raises(ValueError):
        scheduler.validate_scheduler_limits(
            max_iterations=scheduler.MAX_SCHEDULER_ITERATIONS + 1,
            max_steps=1,
            interval_seconds=0,
            lease_ttl_seconds=120,
        )
    with pytest.raises(ValueError):
        scheduler.validate_scheduler_limits(
            max_iterations=1,
            max_steps=1,
            interval_seconds=0,
            lease_ttl_seconds=scheduler.MIN_LEASE_TTL_SECONDS - 1,
        )


def test_scheduler_stops_when_queue_is_empty(monkeypatch):
    calls = {"batches": 0, "released": 0}

    async def acquire(*_args, **_kwargs):
        return True

    async def execute(*_args, **_kwargs):
        calls["batches"] += 1
        return [_outcome()] if calls["batches"] == 1 else []

    async def renew(*_args, **_kwargs):
        return True

    async def release(*_args, **_kwargs):
        calls["released"] += 1

    monkeypatch.setattr(scheduler, "acquire_worker_lease", acquire)
    monkeypatch.setattr(scheduler, "execute_lgbm_steps", execute)
    monkeypatch.setattr(scheduler, "renew_worker_lease", renew)
    monkeypatch.setattr(scheduler, "release_worker_lease", release)

    result = asyncio.run(
        scheduler.run_lgbm_scheduler(
            _Session(),
            1,
            max_iterations=5,
            max_steps=1,
        )
    )

    assert result.status == "COMPLETED"
    assert result.stop_reason == "no_pending_steps"
    assert result.iterations == 2
    assert len(result.outcomes) == 1
    assert calls["released"] == 1


def test_scheduler_stops_on_non_success_outcome(monkeypatch):
    calls = {"released": 0}

    async def acquire(*_args, **_kwargs):
        return True

    async def execute(*_args, **_kwargs):
        return [_outcome("INSUFFICIENT_DATA")]

    async def release(*_args, **_kwargs):
        calls["released"] += 1

    monkeypatch.setattr(scheduler, "acquire_worker_lease", acquire)
    monkeypatch.setattr(scheduler, "execute_lgbm_steps", execute)
    monkeypatch.setattr(scheduler, "release_worker_lease", release)

    result = asyncio.run(
        scheduler.run_lgbm_scheduler(_Session(), 1, max_iterations=3)
    )

    assert result.status == "STOPPED_ON_OUTCOME"
    assert result.stop_reason == "non_success_outcome"
    assert result.iterations == 1
    assert calls["released"] == 1


def test_scheduler_reports_active_lease_without_running_steps(monkeypatch):
    async def acquire(*_args, **_kwargs):
        return False

    async def execute(*_args, **_kwargs):
        raise AssertionError("worker must not run when lease is held")

    monkeypatch.setattr(scheduler, "acquire_worker_lease", acquire)
    monkeypatch.setattr(scheduler, "execute_lgbm_steps", execute)

    result = asyncio.run(scheduler.run_lgbm_scheduler(_Session(), 1))

    assert result.status == "ALREADY_RUNNING"
    assert result.iterations == 0
    assert result.outcomes == ()


def test_scheduler_preserves_completed_outcomes_on_batch_error(monkeypatch):
    async def acquire(*_args, **_kwargs):
        return True

    async def execute(*_args, **_kwargs):
        raise scheduler.ExecutionBatchError(
            RuntimeError("write failed"),
            [_outcome()],
        )

    async def release(*_args, **_kwargs):
        return None

    monkeypatch.setattr(scheduler, "acquire_worker_lease", acquire)
    monkeypatch.setattr(scheduler, "execute_lgbm_steps", execute)
    monkeypatch.setattr(scheduler, "release_worker_lease", release)

    result = asyncio.run(scheduler.run_lgbm_scheduler(_Session(), 1))

    assert result.status == "PARTIAL_FAILURE"
    assert result.stop_reason == "batch_failure"
    assert len(result.outcomes) == 1
