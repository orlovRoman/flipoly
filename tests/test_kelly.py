import pytest
import pickle
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from unittest.mock import MagicMock, AsyncMock, PropertyMock

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, ModelRegistry
from polyflip.trading.utils import compute_kelly_fraction
from polyflip.trading.engine import trade_worker_cycle
from polyflip.config import Settings

class PickleableMockModel:
    def __init__(self):
        self.feature_names_in_ = ["mid_price"]
    def predict_proba(self, X):
        return [[0.95, 0.05]]  # prob_favorite=0.95, prob_flip=0.05

def test_compute_kelly_fraction_strong_signal():
    f = compute_kelly_fraction(p_win=0.70, buy_price=0.30, max_fraction=0.10)
    # edge = 0.70 - 0.30 = 0.40
    # f = 0.40 / 0.70 = 0.57 -> capped at 0.10
    assert f == 0.10

def test_compute_kelly_fraction_weak_signal():
    f = compute_kelly_fraction(p_win=0.40, buy_price=0.50, max_fraction=0.10)
    # edge = 0.40 - 0.50 = -0.10 <= 0
    assert f == 0.0

def test_compute_kelly_fraction_boundary_price():
    f = compute_kelly_fraction(p_win=0.80, buy_price=0.0, max_fraction=0.10)
    assert f == 0.0  # fallback for price <= 0

@pytest.mark.asyncio
async def test_kelly_disabled_uses_fixed_bet(db_session):
    now = datetime.now(timezone.utc)
    
    # 1. Настройки в БД: KELLY_ENABLED="false"
    db_settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="360", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="KELLY_ENABLED", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE", value="0.40", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)

    # 2. Создаем рынок BTC
    market = LiveMarket(
        market_id="m_btc_fixed", asset="BTC", question="BTC Up?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=120),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)

    db_session.add(ModelRegistry(
        asset="BTC", model_blob=pickle.dumps(PickleableMockModel()), is_active=True,
        version=1, accuracy=0.9, features="mid_price", trained_at=now
    ))
    await db_session.commit()

    mock_trader = MagicMock()
    mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None, "executed_usdc": 10.0, "executed_price": 0.58})
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.58})
    mock_api.close = AsyncMock()

    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # Проверяем запись в БД
    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_btc_fixed"))
    trade = res.scalar_one()
    assert trade.amount_usdc == 10.0  # Фиксированная ставка
    assert trade.kelly_fraction is None
    assert trade.kelly_multiplier == 1.0

@pytest.mark.asyncio
async def test_kelly_enabled_scales_bet(db_session):
    now = datetime.now(timezone.utc)
    
    # 1. Настройки в БД: KELLY_ENABLED="true"
    db_settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="360", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="KELLY_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="INITIAL_CAPITAL", value="1000.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="KELLY_MAX_FRACTION", value="0.10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE", value="0.40", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)

    # 2. Создаем рынок BTC
    market = LiveMarket(
        market_id="m_btc_kelly", asset="BTC", question="BTC Up?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=120),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)

    db_session.add(ModelRegistry(
        asset="BTC", model_blob=pickle.dumps(PickleableMockModel()), is_active=True,
        version=1, accuracy=0.9, features="mid_price", trained_at=now
    ))
    await db_session.commit()

    mock_trader = MagicMock()
    mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None, "executed_price": 0.58})
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.58})
    mock_api.close = AsyncMock()

    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # Проверяем аргументы вызова
    args, kwargs = mock_trader.execute_trade.call_args
    actual_size = kwargs.get("size") if kwargs else args[4]
    # capital=1000, kelly_f=0.10 -> bet=100, shares = 100/0.58 ≈ 172.41
    assert abs(actual_size - round(100.0 / 0.58, 2)) < 0.1

    # Проверяем запись в БД
    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_btc_kelly"))
    trade = res.scalar_one()
    # capital = 1000.0, p_win = 0.95, buy_price = 0.58
    # edge = 0.95 - 0.58 = 0.37
    # f = 0.37 / 0.42 = 0.88 -> capped at 0.10
    # expected bet size = 100.0
    assert trade.kelly_fraction == 0.10
    assert trade.amount_usdc == 100.0
