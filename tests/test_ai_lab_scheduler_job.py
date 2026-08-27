from types import SimpleNamespace

import pytest

from polyflip.scheduler import jobs


class _ScalarResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return self

    def all(self):
        return self._values


class _Session:
    async def execute(self, _statement):
        return _ScalarResult([7, 8])


class _SessionContext:
    def __init__(self):
        self.session = _Session()

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False


@pytest.mark.asyncio
async def test_ai_lab_execution_job_drains_planned_runs(monkeypatch):
    contexts = []
    calls = []

    def session_factory():
        context = _SessionContext()
        contexts.append(context)
        return context

    async def run_scheduler(session, run_id, **kwargs):
        assert isinstance(session, _Session)
        calls.append((run_id, kwargs))
        return SimpleNamespace(
            status="COMPLETED",
            outcomes=(object(),),
            stop_reason="no_pending_steps",
        )

    monkeypatch.setattr(jobs, "async_session", session_factory)
    monkeypatch.setattr(jobs, "run_lgbm_scheduler", run_scheduler)

    await jobs.ai_lab_execution_job()

    assert [run_id for run_id, _ in calls] == [7, 8]
    assert all(
        kwargs
        == {
            "max_iterations": 1,
            "max_steps": 5,
            "lease_ttl_seconds": 120,
            "manage_run_lease": False,
        }
        for _, kwargs in calls
    )
    assert len(contexts) == 3
