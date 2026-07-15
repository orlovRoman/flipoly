import pytest
import pickle
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from unittest.mock import patch, AsyncMock

from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory
from polyflip.trading.engine import trade_worker_cycle

class MockModel:
    def __init__(self, proba):
        self.proba = proba
        self.feature_names_in_ = ["mid_price"]
    def predict_proba(self, X):
        return [self.proba]

@pytest.mark.asyncio
async def test_engine_dynamic_no_flip_threshold(db_session):
    # Тестируем, что no_flip_threshold для ассета пересчитывается как (flip_threshold - DEAD_ZONE_WIDTH)
    now = datetime.now(timezone.utc)
    
    # 1. Задаем настройки в БД:
    # Глобальный no_flip = 0.15 (но он должен переопределиться!)
    # Для BTC задаем per-asset flip_threshold = 0.60.
    # Значит, для BTC рекомендованный и применяемый no_flip порог должен стать 0.60 - 0.15 = 0.45.
    db_settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="360", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD_BTC", value="0.60", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)

    # 2. Создаем рынок BTC. У него current_yes_price = 0.6 (YES фаворит).
    market = LiveMarket(
        market_id="m_btc", asset="BTC", question="BTC Up?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=120),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)

    # 3. Модель выдает p_flip = 0.30.
    # Если бы применялся глобальный no_flip (0.15), то p_flip (0.30) > 0.15 -> сделка бы пропустилась.
    # Но так как у нас per-asset порог для BTC = 0.60, то no_flip порог пересчитывается как 0.45.
    # Так как p_flip (0.30) < 0.45, мы встаем ПО ТРЕНДУ (купить фаворита YES).
    # И при p_win = 0.70 и цене 0.62 ставка Келли будет > 0 (f ≈ 21%, ограничена 10% капитала = 20 USDC).
    model = MockModel([0.70, 0.30])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()

    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls:
         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.62})
         mock_api.close = AsyncMock()

         await trade_worker_cycle(db_session, mock_trader, mock_api)

         # 4. Проверяем, что сделка совершена
         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()
         assert len(trades) == 1
         assert trades[0].status == "SUCCESS"
         assert trades[0].outcome_bought == "YES", "Должны были купить фаворита по тренду!"
         assert trades[0].executed_price == 0.62

@pytest.mark.asyncio
async def test_recommended_thresholds_api(db_session):
    from polyflip.api.settings import get_recommended_thresholds
    
    # 1. Задаем настройки в нашей тестовой БД (db_session)
    now = datetime.now(timezone.utc)
    db_settings = [
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.20", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD_BTC", value="0.65", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)
    await db_session.commit()

    # 2. Патчим async_session в settings.py, чтобы он возвращал нашу тестовую сессию db_session
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass

    def dummy_session_creator():
        return DummyAsyncContextManager(db_session)

    from polyflip.config import Settings
    from unittest.mock import PropertyMock

    with patch("polyflip.api.settings.async_session", dummy_session_creator), \
         patch.object(Settings, "asset_list", new_callable=PropertyMock) as mock_prop:
         mock_prop.return_value = ["BTC", "ETH"]
         response = await get_recommended_thresholds()
         
         # 3. Проверяем результаты
         assert response["global"]["dead_zone"] == 0.15
         assert response["global"]["current_no_flip"] == 0.20
         
         assert "BTC" in response["per_asset"]
         assert response["per_asset"]["BTC"]["flip_threshold"] == 0.65
         assert response["per_asset"]["BTC"]["recommended_no_flip"] == 0.50 # 0.65 - 0.15
         assert response["per_asset"]["BTC"]["is_auto_calibrated"] is True
         
         assert "ETH" not in response["per_asset"], "Для ETH порога в БД нет, не должно быть в per_asset"

@pytest.mark.asyncio
async def test_only_favorite_skips_flip_signal(db_session):
    """При высоком p_flip → сделка пропускается со статусом Ожидается флип"""
    now = datetime.now(timezone.utc)
    
    # 1. Задаем настройки в БД:
    db_settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="360", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD_BTC", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)

    # 2. Создаем рынок BTC. У него current_yes_price = 0.3 (аутсайдер).
    # YES фаворит - NO, так как его цена 0.7.
    # Если мы ждем флип (p_flip = 0.90 >= calibrated_val = 0.85), сделка должна пропускаться
    market = LiveMarket(
        market_id="m_btc_fav", asset="BTC", question="BTC Up?",
        current_yes_price=0.3, current_no_price=0.7, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=120),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)

    # Модель выдает p_flip = 0.95 (> upper = 0.925).
    model = MockModel([0.05, 0.95])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()

    from unittest.mock import MagicMock, AsyncMock
    mock_trader = MagicMock()
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.30, "current_spread": 0.01, "best_ask": 0.5})
    mock_api.close = AsyncMock()

    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # 4. Проверяем TradeHistory
    res = await db_session.execute(select(TradeHistory))
    trades = res.scalars().all()
    
    # Должна быть ровно одна запись о пропуске
    assert len(trades) == 1
    assert trades[0].status == "SKIPPED"
    assert "Ожидается флип" in trades[0].error_msg
    assert trades[0].predicted_flip_prob == 0.95
    assert mock_trader.execute_trade.call_count == 0
