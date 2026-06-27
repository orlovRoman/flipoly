import pytest
from unittest.mock import patch, PropertyMock
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, ModelRegistry
from polyflip.trading.utils import compute_kelly_multiplier
from polyflip.trading.engine import trade_worker_cycle
from datetime import timedelta
from sqlalchemy import select
from unittest.mock import MagicMock, AsyncMock
import pickle

class PickleableMockModel:
    def __init__(self):
        self.feature_names_in_ = ["mid_price"]
    def predict_proba(self, X):
        return [[0.95, 0.05]]


def test_compute_kelly_multiplier_strong_signal():
    f, mult = compute_kelly_multiplier(p_win=0.70, buy_price=0.30)
    assert f > 0.0
    assert 1.0 < mult <= 2.0

def test_compute_kelly_multiplier_weak_signal():
    f, mult = compute_kelly_multiplier(p_win=0.40, buy_price=0.50)
    assert f == 0.0
    assert mult == 1.0  # штраф за нулевой edge

def test_compute_kelly_multiplier_boundary_price():
    f, mult = compute_kelly_multiplier(p_win=0.80, buy_price=0.0)
    assert f == 0.0 and mult == 1.0  # деление на 0 → безопасный fallback

def test_compute_kelly_multiplier_max_fraction():
    f, mult = compute_kelly_multiplier(p_win=0.99, buy_price=0.01)
    assert f == 0.10  # зафиксировано max_fraction
    assert mult == 2.0

def test_capital_not_referenced():
    import inspect
    import polyflip.trading.engine as engine_module
    
    source = inspect.getsource(engine_module.trade_worker_cycle)
    lines = [line for line in source.split('\n') if 'capital' in line and '#' not in line]
    assert len(lines) == 0, f"Найдены строки с 'capital' в trade_worker_cycle: {lines}"

def test_kelly_multiplier_range():
    # kelly_f = 0.0  → multiplier = 1.0 (штраф за слабый сигнал)
    # kelly_f = 0.05 → multiplier = 1.5
    # kelly_f = 0.10 → multiplier = 2.0
    for kelly_f, expected in [(0.0, 1.0), (0.05, 1.5), (0.10, 2.0)]:
        mult = 1.0 + (kelly_f / 0.10)
        assert abs(mult - expected) < 0.01

@pytest.mark.asyncio
async def test_kelly_stats_exclude_zero_fraction(db_session):
    from polyflip.api.trading_dashboard import get_trading_stats
    from polyflip.config import Settings

    # Настраиваем INITIAL_CAPITAL
    db_session.add(RuntimeSettings(key="INITIAL_CAPITAL", value="1000", updated_at=datetime.now(timezone.utc), updated_by="test"))
    
    now = datetime.now(timezone.utc)
    # Создаём 1 SUCCESS с kelly_fraction=0.08, kelly_multiplier=1.8
    t1 = TradeHistory(
        market_id="m1", asset="BTC", outcome_bought="YES", amount_usdc=18.0, executed_price=0.5,
        predicted_flip_prob=0.8, active_features="", status="SUCCESS", pnl=10.0,
        kelly_fraction=0.08, kelly_multiplier=1.8, created_at=now
    )
    # Создаём 1 SUCCESS с kelly_fraction=0.0, kelly_multiplier=1.0 (легитимная сделка с нулевым edge)
    t2 = TradeHistory(
        market_id="m2", asset="BTC", outcome_bought="YES", amount_usdc=10.0, executed_price=0.5,
        predicted_flip_prob=0.5, active_features="", status="SUCCESS", pnl=2.0,
        kelly_fraction=0.0, kelly_multiplier=1.0, created_at=now
    )
    # Создаём 5 SKIPPED с kelly_fraction=None, kelly_multiplier=None
    skipped_trades = [
        TradeHistory(
            market_id=f"m_skip_{i}", asset="BTC", outcome_bought="NONE", amount_usdc=0.0, executed_price=0.0,
            predicted_flip_prob=0.5, active_features="", status="SKIPPED", pnl=None,
            kelly_fraction=None, kelly_multiplier=None, created_at=now
        )
        for i in range(5)
    ]

    db_session.add_all([t1, t2])
    db_session.add_all(skipped_trades)
    await db_session.commit()

    with patch.object(Settings, "asset_list", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = ["BTC"]
        stats = await get_trading_stats(db_session)
        
    k = stats["kelly_stats"]
    assert abs(k["avg_f"] - 0.04) < 0.001, f"Ожидали avg_f=0.04, получили {k['avg_f']}"
    assert abs(k["avg_mult"] - 1.4) < 0.01, f"Ожидали avg_mult=1.4, получили {k['avg_mult']}"

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
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="KELLY_ENABLED", value="false", updated_at=now, updated_by="test")
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
    mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"best_ask": 0.42})
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
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_ASSETS", value="BTC", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="KELLY_ENABLED", value="true", updated_at=now, updated_by="test")
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
    mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"best_ask": 0.42})
    mock_api.close = AsyncMock()

    await trade_worker_cycle(db_session, mock_trader, mock_api)

    # Проверяем запись в БД
    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_btc_kelly"))
    trade = res.scalar_one()
    assert trade.kelly_multiplier > 1.0, "Kelly должен увеличить ставку"
    assert trade.amount_usdc > 10.0, "Ставка должна быть больше базовой"
    assert trade.kelly_fraction is not None
