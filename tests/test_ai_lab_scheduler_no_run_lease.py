import asyncio
from types import SimpleNamespace

from polyflip.ai_lab import scheduler
from polyflip.ai_lab.executor import ExecutionOutcome


class _Session:
    def __init__(self):
        self.rollback_count = 0

    async def get(self, _model, _run_id):
        return SimpleNamespace(status="RUNNING")

    async def rollback(self):
        self.rollback_count += 1


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


def test_internal_scheduler_does_not_touch_external_agent_lease(monkeypatch):
    calls = {"batches": 0}

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("internal execution must not manage agent lease")

    async def execute(*_args, **_kwargs):
        calls["batches"] += 1
        return [_outcome()] if calls["batches"] == 1 else []

    monkeypatch.setattr(scheduler, "acquire_worker_lease", forbidden)
    monkeypatch.setattr(scheduler, "renew_worker_lease", forbidden)
    monkeypatch.setattr(scheduler, "release_worker_lease", forbidden)
    monkeypatch.setattr(scheduler, "execute_lgbm_steps", execute)

    result = asyncio.run(
        scheduler.run_lgbm_scheduler(
            _Session(),
            1,
            max_iterations=2,
            max_steps=1,
            manage_run_lease=False,
        )
    )

    assert result.status == "COMPLETED"
    assert result.stop_reason == "no_pending_steps"
    assert len(result.outcomes) == 1
