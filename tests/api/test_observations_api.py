"""
tests/api/test_observations_api.py

Tests for /health/observations API endpoint.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.api.main import app
from polyflip.db.connection import get_db_session
from polyflip.crypto.underlying_observations import Observation, ObservationRepository, get_observation_writer


@pytest.mark.asyncio
async def test_health_observations_endpoint(db_session: AsyncSession):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    repo = ObservationRepository(db_session)
    now = datetime.now(timezone.utc)
    await repo.save(Observation("BTC", 78000.0, "BINANCE", now, now))
    await repo.save(Observation("BTC", 78020.0, "ORACLE", now, now))

    writer = get_observation_writer()
    writer.record_tick("BTC", 78000.0, "BINANCE", event_at=now, received_at=now)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/observations")
            assert resp.status_code == 200
            data = resp.json()
            assert "healthy" in data
            assert data["btc_count"] == 2
            assert data["btc_state"]["instrument"] == "BTC"
            assert data["btc_state"]["binance_price"] == 78000.0
            assert data["btc_state"]["oracle_price"] == 78020.0
            assert data["btc_state"]["status"] == "VALID"
    finally:
        app.dependency_overrides.pop(get_db_session, None)
