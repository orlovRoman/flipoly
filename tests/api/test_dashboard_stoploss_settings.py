import pytest
from httpx import ASGITransport, AsyncClient
from polyflip.api.main import app


@pytest.mark.asyncio
async def test_patch_stop_loss_pct_valid(db_session, monkeypatch):
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass
            
    monkeypatch.setattr("polyflip.api.settings.async_session", lambda: DummyAsyncContextManager(db_session))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Валидный патч
        resp = await client.put("/api/settings/STOP_LOSS_PCT", json={"value": "40.0"}, headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "40.0"


@pytest.mark.asyncio
async def test_patch_stop_loss_pct_invalid(db_session, monkeypatch):
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass
            
    monkeypatch.setattr("polyflip.api.settings.async_session", lambda: DummyAsyncContextManager(db_session))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Инвалидный патч (0%)
        resp = await client.put("/api/settings/STOP_LOSS_PCT", json={"value": "0"}, headers={"X-API-Key": "test-key"})
        assert resp.status_code == 400  # FastAPI выбрасывает 400 из-за HTTPException
        
        # Инвалидный патч (100%)
        resp = await client.put("/api/settings/STOP_LOSS_PCT", json={"value": "100"}, headers={"X-API-Key": "test-key"})
        assert resp.status_code == 400
