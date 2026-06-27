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
async def test_engine_skips_when_trading_disabled(db_session):
    trader_mock = AsyncMock()
    api_client_mock = AsyncMock()
    await trade_worker_cycle(db_session, trader_mock, api_client_mock)
    res = await db_session.execute(select(TradeHistory))
    assert len(res.scalars().all()) == 0

@pytest.mark.asyncio
async def test_engine_makes_trade_outsider(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ONLY_FAVORITE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
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
    
    # Model predicts flip prob = 0.9
    model = MockModel([0.1, 0.9])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"best_ask": 0.42})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         
         assert len(trades) == 1
         # mid_price = 0.6 (YES is fav). p_flip > 0.85 -> buy NO.
         assert trades[0].outcome_bought == "NO"
         assert trades[0].executed_price == 0.42
         
         mock_trainer_call = mock_trader.execute_trade.call_args
         assert mock_trainer_call is not None, "execute_trade was never called"
         _, kwargs = mock_trainer_call
         assert kwargs["price"] == 0.42
         assert kwargs["side"] == "BUY"

@pytest.mark.asyncio
async def test_engine_respects_only_favorite(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ONLY_FAVORITE", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m2", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # Model predicts flip prob = 0.9 (Signal to buy outsider)
    model = MockModel([0.1, 0.9])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"best_ask": 0.42})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         
         # Should contain 1 record with SKIPPED status because only_favorite blocked the outsider bet
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         assert "Only Favorite is enabled" in trades[0].error_msg
         assert mock_trader.execute_trade.call_count == 0

@pytest.mark.asyncio
async def test_save_or_update_no_extra_select(db_session):
    """Функция save_or_update_skipped_trade не должна делать SELECT, если передан existing_skipped."""
    from polyflip.trading.engine import save_or_update_skipped_trade
    
    # Мокаем execute на сессии БД
    original_execute = db_session.execute
    mock_execute = AsyncMock(side_effect=original_execute)
    db_session.execute = mock_execute
    
    class FakeMarket:
        market_id = "m1"
        asset = "BTC"
    
    now = datetime.now(timezone.utc)
    existing = TradeHistory(
        market_id="m1", asset="BTC", outcome_bought="NONE", amount_usdc=0.0,
        executed_price=0.0, predicted_flip_prob=0.5, active_features="",
        model_version=1, status="SKIPPED", error_msg="Old reason", created_at=now
    )
    
    await save_or_update_skipped_trade(
        db_session=db_session,
        market=FakeMarket(),
        reason="New reason",
        p_flip_val=0.6,
        model_version=1,
        start_time=now,
        existing_skipped=existing
    )
    
    # Ни одного execute не должно быть вызвано (объект обновляется прямо в памяти)
    assert mock_execute.call_count == 0
    assert existing.error_msg == "New reason"
    assert existing.predicted_flip_prob == 0.6
