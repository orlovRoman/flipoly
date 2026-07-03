import pytest
import pickle
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from unittest.mock import MagicMock, AsyncMock

from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory
from polyflip.trading.engine import trade_worker_cycle


class DynamicMockModel:
    def __init__(self, prob_yes: float = 0.50):
        self.prob_yes = prob_yes
        self.feature_names_in_ = ["mid_price"]

    def predict_proba(self, X):
        return [[1.0 - self.prob_yes, self.prob_yes]]


@pytest.mark.asyncio
async def test_skipped_then_signal_appears(db_session):
    """Тест: бот пропустил рынок на первом тике (SKIPPED), а на втором появился сильный сигнал — бот должен войти."""
    now = datetime.now(timezone.utc)

    # 1. Настройки
    db_settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="360", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="1.50", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)

    # 2. Создаем рынок
    market = LiveMarket(
        market_id="m_reentry", asset="BTC", question="BTC Up?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        price_velocity=0.0, volume_5min=100.0,
        yes_token_id="tok_yes", no_token_id="tok_no",
        end_time_est=now + timedelta(seconds=120), last_updated=now
    )
    db_session.add(market)

    # 3. Регистрируем модель с начальной вероятностью 0.50
    mock_model = DynamicMockModel(prob_yes=0.50)
    db_session.add(ModelRegistry(
        asset="BTC", model_blob=pickle.dumps(mock_model), is_active=True,
        version=1, accuracy=0.9, features="mid_price", trained_at=now
    ))
    await db_session.commit()

    # Мокаем API и Trader
    mock_trader = MagicMock()
    mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.58})
    mock_api.close = AsyncMock()

    # ---------------------------------------------------------
    # Цикл 1: p_flip = 0.50 (внутри диапазона пропуска) -> SKIPPED
    # ---------------------------------------------------------
    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # Проверяем запись в БД
    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_reentry"))
    trades = res.scalars().all()
    assert len(trades) == 1
    assert trades[0].status == "SKIPPED"
    assert trades[0].predicted_flip_prob == 0.50

    # ---------------------------------------------------------
    # Цикл 2: p_flip = 0.90 (выше порога 0.85) -> SUCCESS
    # ---------------------------------------------------------
    # Обновляем модель в БД новой вероятностью 0.05 (высокая уверенность в фаворите)
    res_reg = await db_session.execute(select(ModelRegistry).where(ModelRegistry.asset == "BTC"))
    registry = res_reg.scalar_one()
    mock_model.prob_yes = 0.05
    registry.model_blob = pickle.dumps(mock_model)
    await db_session.commit()

    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # Проверяем запись в БД: старый SKIPPED должен удалиться, новый SUCCESS записаться
    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_reentry"))
    trades = res.scalars().all()
    assert len(trades) == 1
    assert trades[0].status == "SUCCESS"
    assert trades[0].predicted_flip_prob == 0.05


@pytest.mark.asyncio
async def test_no_double_entry(db_session):
    """Тест: после SUCCESS бот не входит повторно в рынок."""
    now = datetime.now(timezone.utc)

    # 1. Настройки
    db_settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="360", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_NO_FLIP_THRESHOLD", value="0.15", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="ETH", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="1.50", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(db_settings)

    # 2. Создаем рынок
    market = LiveMarket(
        market_id="m_double", asset="ETH", question="ETH Up?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        price_velocity=0.0, volume_5min=100.0,
        yes_token_id="tok_yes", no_token_id="tok_no",
        end_time_est=now + timedelta(seconds=120), last_updated=now
    )
    db_session.add(market)

    # 3. Регистрируем модель с вероятностью 0.05
    mock_model = DynamicMockModel(prob_yes=0.05)
    db_session.add(ModelRegistry(
        asset="ETH", model_blob=pickle.dumps(mock_model), is_active=True,
        version=1, accuracy=0.9, features="mid_price", trained_at=now
    ))
    await db_session.commit()

    # Мокаем API и Trader
    mock_trader = MagicMock()
    mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.58})
    mock_api.close = AsyncMock()

    # ---------------------------------------------------------
    # Цикл 1: p_flip = 0.90 -> SUCCESS
    # ---------------------------------------------------------
    await trade_worker_cycle(db_session, mock_trader, mock_api)

    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_double"))
    trades = res.scalars().all()
    assert len(trades) == 1
    assert trades[0].status == "SUCCESS"

    # ---------------------------------------------------------
    # Цикл 2: p_flip = 0.95 (еще сильнее сигнал) -> Должен пропустить, так как уже зашли
    # ---------------------------------------------------------
    # Обновляем модель в БД новой вероятностью 0.02 (еще сильнее сигнал)
    res_reg = await db_session.execute(select(ModelRegistry).where(ModelRegistry.asset == "ETH"))
    registry = res_reg.scalar_one()
    mock_model.prob_yes = 0.02
    registry.model_blob = pickle.dumps(mock_model)
    await db_session.commit()

    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # В базе должна остаться ровно ОДНА запись со статусом SUCCESS
    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_double"))
    trades = res.scalars().all()
    assert len(trades) == 1
    assert trades[0].status == "SUCCESS"
    assert trades[0].predicted_flip_prob == 0.05  # значение не обновилось

    # Трейдер должен быть вызван ровно один раз за все время
    assert mock_trader.execute_trade.call_count == 1
