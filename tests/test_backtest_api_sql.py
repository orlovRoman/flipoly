import pytest
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient, ASGITransport
from polyflip.api.main import app
from polyflip.db.models import MarketSnapshot
from polyflip.api.backtest_api import get_db_session

class DummyAsyncContextManager:
    def __init__(self, session):
        self.session = session
    async def __aenter__(self):
        return self.session
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_backtest_sql_correctness(db_session):
    # Override get_db_session dependency to use test db_session
    app.dependency_overrides[get_db_session] = lambda: db_session

    now = datetime.now(timezone.utc)
    # Market 1: has 3 snapshots, in window [1, 60]
    # Market 2: has 1 snapshot, in window
    # Market 3: has 5 snapshots, but only 1 inside the window [1, 60] (others are 70, 80, 90, 100)
    snaps = [
        # Market 1
        MarketSnapshot(
            asset="BTC", market_id="m1", time_left_min=45.0, mid_price=0.6,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="YES", flip_vs_final=False, recorded_at=now - timedelta(minutes=45)
        ),
        MarketSnapshot(
            asset="BTC", market_id="m1", time_left_min=30.0, mid_price=0.7,
            spread=0.02, volume_5min=20.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="YES", flip_vs_final=False, recorded_at=now - timedelta(minutes=30)
        ),
        MarketSnapshot(
            asset="BTC", market_id="m1", time_left_min=15.0, mid_price=0.8,
            spread=0.02, volume_5min=30.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="YES", flip_vs_final=False, recorded_at=now - timedelta(minutes=15)
        ),
        # Market 2
        MarketSnapshot(
            asset="BTC", market_id="m2", time_left_min=30.0, mid_price=0.5,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="YES", flip_vs_final=False, recorded_at=now - timedelta(minutes=30)
        ),
        # Market 3
        MarketSnapshot(
            asset="BTC", market_id="m3", time_left_min=5.0, mid_price=0.6,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO", flip_vs_final=False, recorded_at=now - timedelta(minutes=5)
        ),
        MarketSnapshot(
            asset="BTC", market_id="m3", time_left_min=75.0, mid_price=0.6,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO", flip_vs_final=False, recorded_at=now - timedelta(minutes=75)
        ),
        MarketSnapshot(
            asset="BTC", market_id="m3", time_left_min=90.0, mid_price=0.6,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO", flip_vs_final=False, recorded_at=now - timedelta(minutes=90)
        ),
        MarketSnapshot(
            asset="BTC", market_id="m3", time_left_min=105.0, mid_price=0.6,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO", flip_vs_final=False, recorded_at=now - timedelta(minutes=105)
        ),
        MarketSnapshot(
            asset="BTC", market_id="m3", time_left_min=120.0, mid_price=0.6,
            spread=0.02, volume_5min=10.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO", flip_vs_final=False, recorded_at=now - timedelta(minutes=120)
        ),
    ]
    db_session.add_all(snaps)
    await db_session.commit()

    from unittest.mock import patch

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-API-Key": "test-key"}
        # Run 1: min_snapshots_per_market = 3. Only Market 1 should pass.
        # Market 2 has only 1 snapshot in window.
        # Market 3 has 5 snapshots total, but only 1 is in the window [1, 60]. So it has only 1 in window, filtered out.
        # Total markets in window = 3 (m1, m2, m3).
        # Tradeable = 1 (m1).
        # Skipped should be 2 (m2, m3).
        payload = {
            "assets": ["BTC"],
            "min_snapshots_per_market": 3,
            "min_time_left_min": 1,
            "max_time_left_min": 60,
            "capital": 1000,
            "min_bet": 5,
            "max_bet": 50,
            "strategy_mode": "PURE_FAVORITE",
        }
        with patch("polyflip.api.backtest_api.async_session", return_value=DummyAsyncContextManager(db_session)):
            resp = await client.post("/api/backtest/submit", json=payload, headers=headers)
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]
            
            status_resp = await client.get(f"/api/backtest/status/{run_id}", headers=headers)
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["status"] == "completed", f"Job failed: {data.get('error')}"
            assert data["result"]["tradeable_markets"] == 1
            assert data["result"]["skipped_markets"] == 2

            # Run 2: min_snapshots_per_market = 1.
            # All 3 markets (m1, m2, m3) should pass.
            # Skipped should be 0.
            payload["min_snapshots_per_market"] = 1
            resp = await client.post("/api/backtest/submit", json=payload, headers=headers)
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]
            
            status_resp = await client.get(f"/api/backtest/status/{run_id}", headers=headers)
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["status"] == "completed", f"Job failed: {data.get('error')}"
            assert data["result"]["tradeable_markets"] == 3
            assert data["result"]["skipped_markets"] == 0

    app.dependency_overrides.clear()
