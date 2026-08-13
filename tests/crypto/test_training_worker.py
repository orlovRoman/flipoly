import pytest


@pytest.mark.asyncio
async def test_finish_job_persists_error_and_traceback(monkeypatch):
    from polyflip.crypto import training_worker

    captured = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, statement):
            captured["statement"] = statement

        async def commit(self):
            captured["committed"] = True

    monkeypatch.setattr(training_worker, "async_session", lambda: FakeSession())

    class FakeUpdate:
        def where(self, *conditions):
            captured["conditions"] = conditions
            return self

        def values(self, **values):
            captured["values"] = values
            return self

    monkeypatch.setattr(training_worker, "update", lambda _model: FakeUpdate())

    await training_worker._finish_job(
        42,
        success=False,
        error="training failed",
        error_traceback="Traceback (most recent call last):\\n...",
    )

    assert captured["values"]["status"] == "FAILED"
    assert captured["values"]["error"] == "training failed"
    assert captured["values"]["error_traceback"].startswith("Traceback")
    assert captured["committed"] is True
