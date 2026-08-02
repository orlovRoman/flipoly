import pytest
from polyflip.api.main import app
from polyflip.db.models import TradeHistory, LiveMarket

@pytest.mark.asyncio
async def test_gamma_api_error_returns_503(db_session, monkeypatch):
    from polyflip.api.auth import verify_api_key
    from polyflip.db.connection import get_db_session
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        market = LiveMarket(market_id="test-gamma-err", asset="ETH", question="Q", yes_token_id="1", no_token_id="2", end_time_est=now, current_yes_price=0.5, current_no_price=0.5, current_spread=0.01, last_updated=now, resolution_status="PENDING", trading_status="TRADABLE", accepting_orders=True)
        db_session.add(market)
        trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-gamma-err", position_status="OPEN", remaining_shares=10, outcome_bought="YES", amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="{}", status="SUCCESS", created_at=now)
        db_session.add(trade)
        await db_session.commit()

        import polyflip.execution.live_settlement_service as lss
        async def mock_fetch(*args):
            raise lss.GammaApiError("HTTP 503")
        monkeypatch.setattr(lss, "fetch_polymarket_market", mock_fetch)

        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(f"/api/execution/live/positions/{trade.id}/reconcile-resolution")
        
        assert res.status_code == 503
        
        from sqlalchemy import select
        trade_after = (await db_session.execute(select(TradeHistory).where(TradeHistory.id == trade.id))).scalar_one()
        assert trade_after.position_status == "OPEN"
    finally:
        app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_open_market_syncs_to_tradable(db_session, monkeypatch):
    from polyflip.api.auth import verify_api_key
    from polyflip.db.connection import get_db_session
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        market = LiveMarket(market_id="test-sync-market", asset="ETH", question="Q", yes_token_id="1", no_token_id="2", end_time_est=now, current_yes_price=0.5, current_no_price=0.5, current_spread=0.01, last_updated=now, resolution_status="PENDING", trading_status="UNKNOWN", accepting_orders=None)
        db_session.add(market)
        trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-sync-market", position_status="OPEN", remaining_shares=10, outcome_bought="YES", amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="{}", status="SUCCESS", created_at=now)
        db_session.add(trade)
        await db_session.commit()

        import polyflip.execution.live_settlement_service as lss
        async def mock_fetch(*args):
            return {"closed": False, "acceptingOrders": True}
        monkeypatch.setattr(lss, "fetch_polymarket_market", mock_fetch)

        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(f"/api/execution/live/positions/{trade.id}/reconcile-resolution")
        
        assert res.status_code == 409
        
        from sqlalchemy import select
        market_after = (await db_session.execute(select(LiveMarket).where(LiveMarket.market_id == "test-sync-market"))).scalar_one()
        assert market_after.trading_status == "TRADABLE"
        assert market_after.accepting_orders is True
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            close_res = await client.post(f"/api/execution/positions/{trade.id}/close")
        assert close_res.status_code == 200
        
        from polyflip.db.execution_models import ExecutionRequest
        reqs = (await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id))).scalars().all()
        assert len(reqs) == 1
    finally:
        app.dependency_overrides.clear()
