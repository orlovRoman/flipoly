import pytest
import pickle
from unittest.mock import patch, AsyncMock
from sqlalchemy import select
from datetime import datetime, timezone, timedelta

from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory
from polyflip.trading.engine import trade_worker_cycle

class MockModel:
    def __init__(self, proba):
        self.proba = proba
        self.feature_names_in_ = ["mid_price"]
    def predict_proba(self, X):
        return [self.proba] # [[p_no, p_yes_flip]]

@pytest.mark.asyncio
async def test_no_trade_when_flip_threshold_met(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ON_FLIP", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="FLIP_THRESHOLD", value="0.70", updated_at=now, updated_by="test"),
        RuntimeSettings(key="OUTSIDER_MAX_PRICE", value="0.60", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="2.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE_FILTER", value="2.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_PRICE_DRIFT", value="0.20", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m1", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # Model predicts flip prob = 0.75 (>= FLIP_THRESHOLD=0.70)
    model = MockModel([0.25, 0.75])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "executed_usdc": 10.0, "executed_price": 0.32, "mode": "PAPER"})
         
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(side_effect=lambda token_id: {
             "current_yes_price": 0.60,
             "current_spread": 0.01,
             "best_ask": 0.61,
             "best_bid": 0.59
         } if token_id == "t_yes" else {
             "best_ask": 0.32,
             "best_bid": 0.30
         })
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         assert len(trades) == 1
         assert trades[0].status == "SUCCESS", f"Trade was skipped: {trades[0].error_msg}"
         assert trades[0].outcome_bought == "NO"
         assert trades[0].amount_usdc == 10.0
         assert trades[0].executed_price == 0.32
         assert trades[0].predicted_flip_prob == 0.75

@pytest.mark.asyncio
async def test_no_trade_skipped_when_price_exceeds_max(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ON_FLIP", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="FLIP_THRESHOLD", value="0.70", updated_at=now, updated_by="test"),
        RuntimeSettings(key="OUTSIDER_MAX_PRICE", value="0.60", updated_at=now, updated_by="test"),
        RuntimeSettings(key="NO_MIN_EDGE", value="0.04", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="2.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE_FILTER", value="2.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m1", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # Model predicts flip prob = 0.75
    model = MockModel([0.25, 0.75])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         
         mock_trader = mock_trader_cls.return_value
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(side_effect=lambda token_id: {
             "current_yes_price": 0.60,
             "current_spread": 0.01,
             "best_ask": 0.61,
             "best_bid": 0.59
         } if token_id == "t_yes" else {
             "best_ask": 0.65,
             "best_bid": 0.63
         })
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         assert "max_outsider_price" in trades[0].error_msg or "Price drift" in trades[0].error_msg

@pytest.mark.asyncio
async def test_no_trade_skipped_when_flip_prob_too_small(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ON_FLIP", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD_BTC", value="0.70", updated_at=now, updated_by="test"),
        RuntimeSettings(key="FLIP_THRESHOLD", value="0.70", updated_at=now, updated_by="test"),
        RuntimeSettings(key="OUTSIDER_MAX_PRICE", value="0.75", updated_at=now, updated_by="test"),
        RuntimeSettings(key="NO_MIN_EDGE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="2.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE_FILTER", value="2.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MIN_EDGE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_PRICE_DRIFT", value="0.20", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m1", asset="BTC", question="Test?",
        current_yes_price=0.70, current_no_price=0.30, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # DEAD_ZONE_WIDTH=0.05, FLIP_THRESHOLD=0.70, AUTO_DEAD_ZONE=false
    # => compute_dead_zone(0.70, 0.05, auto_mode=False) => lower=0.65, upper=0.70
    # p_flip=0.67 попадает в мертвую зону [0.65, 0.70] — сделка должна быть пропущена
    model = MockModel([0.33, 0.67])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(side_effect=lambda token_id: {
             "current_yes_price": 0.70,
             "current_spread": 0.01,
             "best_ask": 0.71,
             "best_bid": 0.69
         } if token_id == "t_yes" else {
             "best_ask": 0.31,
             "best_bid": 0.29
         })
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         err = trades[0].error_msg.lower()
         assert "dead" in err or "zone" in err or "мёртв" in err or "зона" in err, (
             f"Ожидалось сообщение о мёртвой зоне, получено: {trades[0].error_msg!r}"
         )
