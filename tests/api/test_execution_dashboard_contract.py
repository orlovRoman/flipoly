import pytest

from polyflip.api.main import app
from polyflip.api.execution_api import get_live_trading_status


def test_trading_pnl_markers_route_matches_frontend_contract():
    paths = set(app.openapi()["paths"])

    assert "/api/trading/pnl-markers" in paths
    assert "/pnl-markers" in paths

from polyflip.api.auth import verify_api_key
app.dependency_overrides[verify_api_key] = lambda: True

@pytest.mark.asyncio
async def test_execution_status_disables_live_switch_in_paper(
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")

    result = await get_live_trading_status(db_session)

    assert result["execution_mode"] == "PAPER"
    assert result["kill_switch_available"] is False

from polyflip.db.models import TradeHistory, LiveMarket

@pytest.mark.asyncio
async def test_close_rejects_resolved_market(db_session):
    from polyflip.api.auth import verify_api_key
    from polyflip.db.connection import get_db_session
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        market = LiveMarket(market_id="test-m", asset="ETH", question="Q", yes_token_id="1", no_token_id="2", end_time_est=now, current_yes_price=0.5, current_no_price=0.5, current_spread=0.01, last_updated=now, resolution_status="RESOLVED", trading_status="CLOSED", accepting_orders=False)
        db_session.add(market)
        trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-m", position_status="OPEN", remaining_shares=10, outcome_bought="YES", amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="{}", status="SUCCESS", created_at=now)
        db_session.add(trade)
        await db_session.commit()

        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(f"/api/execution/positions/{trade.id}/close")
            
        assert res.status_code == 409
        
        # Check no ExecutionRequest created
        from polyflip.db.execution_models import ExecutionRequest
        from sqlalchemy import select
        reqs = (await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id))).scalars().all()
        assert len(reqs) == 0
    finally:
        app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_dashboard_contract(db_session):
    from polyflip.api.auth import verify_api_key
    from polyflip.db.connection import get_db_session
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        market = LiveMarket(market_id="test-m2", asset="ETH", question="Q", yes_token_id="3", no_token_id="4", end_time_est=now, current_yes_price=0.5, current_no_price=0.5, current_spread=0.01, last_updated=now, resolution_status="PENDING", trading_status="TRADABLE", accepting_orders=True)
        db_session.add(market)
        trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-m2", position_status="OPEN", remaining_shares=10, outcome_bought="YES", amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="{}", status="SUCCESS", created_at=now)
        db_session.add(trade)
        
        trade_failed = TradeHistory(mode="LIVE", asset="ETH", market_id="test-m2", position_status="ENTRY_FAILED", remaining_shares=0, outcome_bought="YES", amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="{}", status="FAILED", created_at=now)
        db_session.add(trade_failed)
        await db_session.commit()

        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/api/execution/live/dashboard")
        
        assert res.status_code == 200
        data = res.json()
        assert "positions" in data
        assert "tradable" in data["positions"]
        assert "archive" in data["positions"]
        
        tradable_ids = [p["id"] for p in data["positions"]["tradable"]]
        archive_ids = [p["id"] for p in data["positions"]["archive"]]
        
        assert trade.id in tradable_ids
        assert trade_failed.id in archive_ids
        assert trade_failed.id not in tradable_ids
        
        pos = data["positions"]["tradable"][0]
        assert "available_actions" in pos
        assert "redemption_status" in pos
        assert pos["available_actions"]["redeem"] is False
    finally:
        app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_redeem_returns_501(db_session):
    from polyflip.api.auth import verify_api_key
    from polyflip.db.connection import get_db_session
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-m3", position_status="RESOLVED_REDEEMABLE", remaining_shares=10, outcome_bought="YES", amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="{}", status="SUCCESS", created_at=now)
        db_session.add(trade)
        await db_session.commit()

        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(f"/api/execution/live/positions/{trade.id}/redeem")
        
        assert res.status_code == 501
    finally:
        app.dependency_overrides.clear()

