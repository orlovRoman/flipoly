import pytest
from decimal import Decimal
import datetime
from polyflip.execution.live_settlement_service import (
    ResolutionResult,
    parse_confirmed_resolution,
    reconcile_live_resolution,
    LivePositionNotFound,
    MarketNotResolved,
)
from polyflip.db.models import TradeHistory, LiveMarket

def test_parse_confirmed_resolution():
    market_pending = {"closed": False, "active": True}
    res = parse_confirmed_resolution(market_pending)
    assert res is None

    market_resolved_yes = {
        "closed": True,
        "active": False,
        "winning_outcome": "Yes"
    }
    res = parse_confirmed_resolution(market_resolved_yes)
    assert res is not None
    assert res.final_outcome == "YES"

    market_resolved_no = {
        "closed": True,
        "active": False,
        "winning_outcome": "No"
    }
    res = parse_confirmed_resolution(market_resolved_no)
    assert res is not None
    assert res.final_outcome == "NO"

    market_invalid = {
        "closed": True,
        "active": False,
        "winning_outcome": "INVALID"
    }
    res = parse_confirmed_resolution(market_invalid)
    assert res is not None
    assert res.final_outcome == "INVALID"
    
@pytest.mark.asyncio
async def test_reconcile_live_resolution(monkeypatch, db_session):
    async def mock_fetch(market_id):
        return {
            "closed": True,
            "active": False,
            "winning_outcome": "YES"
        }
    monkeypatch.setattr("polyflip.execution.live_settlement_service.fetch_polymarket_market", mock_fetch)

    trade = TradeHistory(
        market_id="0x123",
        asset="TEST",
        mode="LIVE",
        position_status="OPEN",
        outcome_bought="YES",
        amount_usdc=Decimal("5.0"),
        executed_price=Decimal("0.5"),
        predicted_flip_prob=0.5,
        status="FILLED",
        strategy_type="TEST",
        active_features="{}",
        model_version="v1",
        pnl=0.0,
        edge=0.0,
        p_up=0.5,
        strike=0.5,
        market_role="taker",
        remaining_shares=Decimal("10.0"),
        entry_cost_usdc=Decimal("5.0"),
        model_key="test",
        confirm_model_key="test",
        confirm_model_version="test",
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    db_session.add(trade)
    
    market = LiveMarket(
        market_id="0x123",
        asset="TEST",
        question="TEST",
        yes_token_id="1",
        no_token_id="2",
        end_time_est=datetime.datetime.utcnow(),
        last_updated=datetime.datetime.utcnow(),
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.0,
        trading_status="TRADABLE",
        resolution_status="PENDING",
        accepting_orders=True
    )
    db_session.add(market)
    await db_session.commit()

    updated_trade = await reconcile_live_resolution(db_session, trade.id)

    assert updated_trade.position_status == "RESOLVED_REDEEMABLE"
    assert updated_trade.redemption_status == "PENDING"
    assert updated_trade.realized_pnl_usdc == Decimal("5.0")
    
    await db_session.refresh(market)
    assert market.trading_status == "RESOLVED"
    assert market.resolution_status == "RESOLVED"
    assert market.final_outcome == "YES"
    assert market.accepting_orders is False

@pytest.mark.asyncio
async def test_reconcile_live_resolution_loss(monkeypatch, db_session):
    async def mock_fetch(market_id):
        return {
            "closed": True,
            "active": False,
            "winning_outcome": "NO"
        }
    monkeypatch.setattr("polyflip.execution.live_settlement_service.fetch_polymarket_market", mock_fetch)

    trade = TradeHistory(
        market_id="0x456",
        asset="TEST",
        mode="LIVE",
        position_status="OPEN",
        outcome_bought="YES",
        amount_usdc=Decimal("5.0"),
        executed_price=Decimal("0.5"),
        predicted_flip_prob=0.5,
        status="FILLED",
        strategy_type="TEST",
        active_features="{}",
        model_version="v1",
        pnl=0.0,
        edge=0.0,
        p_up=0.5,
        strike=0.5,
        market_role="taker",
        remaining_shares=Decimal("10.0"),
        entry_cost_usdc=Decimal("5.0"),
        model_key="test",
        confirm_model_key="test",
        confirm_model_version="test",
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    db_session.add(trade)
    await db_session.commit()

    updated_trade = await reconcile_live_resolution(db_session, trade.id)

    assert updated_trade.position_status == "RESOLVED_LOST"
    assert updated_trade.redemption_status == "NOT_REQUIRED"
    assert updated_trade.realized_pnl_usdc == Decimal("-5.0")
    assert updated_trade.expected_payout_usdc == Decimal("0")
