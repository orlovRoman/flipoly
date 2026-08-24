import pytest
import uuid
from decimal import Decimal
from httpx import ASGITransport, AsyncClient

from polyflip.api.main import app
from polyflip.db.connection import get_db_session
from polyflip.db.execution_models import LiveTradingSession


@pytest.mark.asyncio
async def test_draft_readiness_without_live_worker_returns_result(db_session):
    """Тест: проверка готовности сессии DRAFT без работающего LIVE воркера возвращает 200 OK с деталями неготовности."""
    app.dependency_overrides[get_db_session] = lambda: db_session

    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="DRAFT",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        filled_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("1.10"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)
    await db_session.commit()

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/execution/live/sessions/{session_obj.id}/readiness",
                headers={"X-API-Key": "test-key"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is False
    assert body["session"]["status"] == "DRAFT"
    assert body["checks"]["live_worker"] is False
    assert body["errors"]


@pytest.mark.asyncio
async def test_security_setting_update_does_not_require_ws_manager(db_session, monkeypatch):
    """Тест: обновление защитной настройки BYPASS_BET_SIZE_CHECK проходит успешно без ws_manager."""
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr("polyflip.api.settings.async_session", lambda: DummyAsyncContextManager(db_session))
    app.dependency_overrides[get_db_session] = lambda: db_session

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/settings/security/BYPASS_BET_SIZE_CHECK",
                json={"value": "false"},
                headers={"X-API-Key": "test-key"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Security setting updated"
    assert data["key"] == "BYPASS_BET_SIZE_CHECK"
    assert data["value"] == "false"
