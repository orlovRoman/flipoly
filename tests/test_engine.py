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
async def test_engine_enters_on_confident_favorite(db_session):
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
        RuntimeSettings(key="MAX_EDGE", value="0.40", updated_at=now, updated_by="test"),
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
    
    # Model predicts flip prob = 0.10 (confident favorite YES)
    model = MockModel([0.9, 0.1])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.61})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         
         assert len(trades) == 1
         # mid_price = 0.6 (YES is fav). p_flip = 0.10 < 0.15 -> buy YES.
         assert trades[0].outcome_bought == "YES"
         assert trades[0].executed_price == 0.61
         assert trades[0].status == "SUCCESS"

@pytest.mark.asyncio
async def test_engine_skips_in_dead_zone(db_session):
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
        RuntimeSettings(key="MAX_EDGE", value="0.40", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
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
    
    # Model predicts flip prob = 0.20 (dead zone [0.15 - 0.30])
    model = MockModel([0.8, 0.2])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.61})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         assert "Мёртвая зона" in trades[0].error_msg
         assert mock_trader.execute_trade.call_count == 0

@pytest.mark.asyncio
async def test_engine_skips_on_high_flip_risk(db_session):
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
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m3", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # Model predicts flip prob = 0.40 (expected flip >= 0.30)
    model = MockModel([0.6, 0.4])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.61})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         assert "Ожидается флип" in trades[0].error_msg
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


@pytest.mark.asyncio
async def test_engine_skips_when_no_fresh_prices(db_session):
    """При отсутствии цен от API движок должен записывать пропуск (SKIPPED)."""
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
        market_id="m_no_price", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # Модель предсказывает низкий риск флипа (0.05)
    model = MockModel([0.95, 0.05])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock()
         mock_api = mock_api_cls.return_value
         # API возвращает пустой словарь (нет свежих цен)
         mock_api.get_market_prices = AsyncMock(return_value={})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
                  
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         assert "No fresh YES prices" in trades[0].error_msg
         assert mock_trader.execute_trade.call_count == 0


@pytest.mark.asyncio
async def test_engine_skips_when_clob_error(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.02", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MIN_EDGE", value="0.05", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)

    market = LiveMarket(
        market_id="m_clob_err", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)

    model = MockModel([0.95, 0.05])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()

    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock()
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"error": "API HTTP Error 429"})
         mock_api.close = AsyncMock()

         await trade_worker_cycle(db_session, mock_trader, mock_api)

         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         
         target_trade = next(t for t in trades if t.market_id == "m_clob_err")
         assert target_trade.status == "SKIPPED"
         assert "API HTTP Error 429" in target_trade.error_msg
         assert mock_trader.execute_trade.call_count == 0


@pytest.mark.asyncio
async def test_engine_skips_when_edge_too_small(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.50", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MIN_EDGE", value="0.05", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m_edge", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # p_flip = 0.40 -> p_win = 0.60
    model = MockModel([0.60, 0.40])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock()
         mock_api = mock_api_cls.return_value
         # buy_price = 0.58. Edge = 0.60 - 0.58 = 0.02 < 0.05
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.58})
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
                  
         assert len(trades) == 1
         assert trades[0].status == "SKIPPED"
         assert "Edge out of bounds" in trades[0].error_msg
         assert abs(trades[0].edge - 0.02) < 1e-4
         assert mock_trader.execute_trade.call_count == 0


@pytest.mark.asyncio
async def test_engine_skips_no_deal_when_edge_too_small(db_session):
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.70", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MIN_EDGE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE", value="0.40", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m_edge_no", asset="BTC", question="Test?",
        current_yes_price=0.3, current_no_price=0.7, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    # p_flip = 0.60 -> p_win = 0.40
    model = MockModel([0.40, 0.60])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock()
         mock_api = mock_api_cls.return_value
         
         # Мокаем get_market_prices: первый раз для YES, второй для NO
         mock_api.get_market_prices = AsyncMock(side_effect=[
             {"current_yes_price": 0.30, "current_spread": 0.01, "best_ask": 0.35},  # YES
             {"current_yes_price": 0.70, "current_spread": 0.01, "best_ask": 0.68}   # NO
         ])
         mock_api.close = AsyncMock()
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
                  
         # Ищем сделку с market_id = "m_edge_no"
         target_trade = next(t for t in trades if t.market_id == "m_edge_no")
         assert target_trade.status == "SKIPPED"
         assert "Edge out of bounds" in target_trade.error_msg
         # edge = p_win (0.4) - buy_price (0.68) = -0.28
         assert abs(target_trade.edge - (-0.28)) < 1e-4
         assert mock_trader.execute_trade.call_count == 0
