import pytest

from polyflip.api.main import app
from polyflip.api.execution_api import get_live_trading_status


def test_trading_pnl_markers_route_matches_frontend_contract():
    paths = {route.path for route in app.routes}

    assert "/api/trading/pnl-markers" in paths
    assert "/pnl-markers" in paths


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
    import datetime
    market = LiveMarket(market_id="test-m", asset="ETH", question="Q", yes_token_id="1", no_token_id="2", end_time_est=datetime.datetime.now(datetime.timezone.utc), current_yes_price=0.5, current_no_price=0.5, current_spread=0.01, resolution_status="RESOLVED", trading_status="CLOSED", accepting_orders=False)
    db_session.add(market)
    trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-m", position_status="OPEN", remaining_shares=10)
    db_session.add(trade)
    await db_session.commit()

    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as client:
        res = await client.post(f"/positions/{trade.id}/close")
        
    assert res.status_code == 409
    
    # check no execution request created
    from polyflip.db.models import ExecutionRequest
    from sqlalchemy import select
    reqs = (await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.trade_id == trade.id))).scalars().all()
    assert len(reqs) == 0

@pytest.mark.asyncio
async def test_dashboard_contract(db_session):
    import datetime
    market = LiveMarket(market_id="test-m2", asset="ETH", question="Q", yes_token_id="3", no_token_id="4", end_time_est=datetime.datetime.now(datetime.timezone.utc), current_yes_price=0.5, current_no_price=0.5, current_spread=0.01, resolution_status="PENDING", trading_status="TRADABLE", accepting_orders=True)
    db_session.add(market)
    trade = TradeHistory(mode="LIVE", asset="ETH", market_id="test-m2", position_status="OPEN", remaining_shares=10)
    db_session.add(trade)
    await db_session.commit()

    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as client:
        res = await client.get(f"/positions?mode=LIVE")
    
    assert res.status_code == 200
    data = res.json()
    assert "positions" in data
    assert "tradable" in data["positions"]
    
    pos = data["positions"]["tradable"][0]
    assert "available_actions" in pos
    assert "redemption_status" in pos

