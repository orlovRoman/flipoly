import pytest
import pickle
from unittest.mock import patch, AsyncMock
from sqlalchemy import select
from datetime import datetime, timezone, timedelta
from sklearn.model_selection import GroupShuffleSplit

from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory, MarketSnapshot
from polyflip.trading.engine import trade_worker_cycle
from polyflip.scheduler.jobs import retrain_job, resolve_trades_job
from polyflip.models.trainer import ModelTrainer

class MockModel:
    def __init__(self, proba):
        self.proba = proba
        self.feature_names_in_ = ["mid_price"]
    def predict_proba(self, X):
        return [self.proba] # [[p_no, p_yes_flip]]

@pytest.mark.asyncio
async def test_bug_01_zero_threshold_ignored(db_session):
    """
    Тест BUG-01: Установка индивидуального порога актива в "0" или "0.0"
    должна игнорироваться, и бот должен использовать глобальный/авто порог,
    вместо того чтобы сбрасывать реальный порог в 0.0 (что блокирует все тренды).
    """
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="0.50", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MIN_EDGE", value="-0.10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="FAVORITE_MIN_EDGE", value="-0.10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE_FILTER", value="1.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD_BTC", value="0", updated_at=now, updated_by="test"), # BUG-01: Наш нуль
    ]
    db_session.add_all(settings)
    
    market = LiveMarket(
        market_id="m_bug1", asset="BTC", question="Test?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    model = MockModel([0.9, 0.1])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    trader_mock = AsyncMock()
    trader_mock.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})
    
    api_client_mock = AsyncMock()
    api_client_mock.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.605})
    
    await trade_worker_cycle(db_session, trader_mock, api_client_mock)
    
    res = await db_session.execute(select(TradeHistory))
    trades = res.scalars().all()
    assert len(trades) == 1
    assert trades[0].status == "SUCCESS"
    assert trades[0].outcome_bought == "YES"  # Бот зашел на фаворита!

@pytest.mark.asyncio
async def test_bug_02_auto_threshold_keys_loaded(db_session):
    """
    Тест BUG-02: Движок должен корректно загружать из БД
    индивидуальные фазовые пороги вида AUTO_FLIP_THRESHOLD_BTC_decided.
    """
    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_EXECUTION_TIME_SEC", value="30", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="ACTIVE_FEATURES", value="mid_price", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="0.50", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MIN_EDGE", value="-0.10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="FAVORITE_MIN_EDGE", value="-0.10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE_FILTER", value="1.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_FLIP_THRESHOLD", value="0.85", updated_at=now, updated_by="test"),
        # BUG-02 FIX: Устанавливаем фазовый порог для decided (mid_price=0.8 -> dev=0.3 -> decided)
        RuntimeSettings(key="AUTO_FLIP_THRESHOLD_BTC_decided", value="0.20", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)
    
    # mid_price = 0.8 -> фаза decided
    market = LiveMarket(
        market_id="m_bug2", asset="BTC", question="Test?",
        current_yes_price=0.8, current_no_price=0.2, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=30),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)
    
    model = MockModel([0.7, 0.3])
    db_session.add(ModelRegistry(asset="BTC", model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features="mid_price", trained_at=now))
    await db_session.commit()
    
    trader_mock = AsyncMock()
    api_client_mock = AsyncMock()
    api_client_mock.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.80, "current_spread": 0.01, "best_ask": 0.805})
    
    await trade_worker_cycle(db_session, trader_mock, api_client_mock)
    
    res = await db_session.execute(select(TradeHistory))
    trades = res.scalars().all()
    assert len(trades) == 1
    assert trades[0].status == "SKIPPED"
    assert "Ожидается флип" in trades[0].error_msg
    assert "0.30" in trades[0].error_msg

@pytest.mark.asyncio
async def test_bug_04_retrain_job_uses_db_assets(db_session):
    """
    Тест BUG-04: retrain_job должен использовать список активов из БД (TRADE_ASSETS),
    а не хардкод из constants.py / settings.py.
    """
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="TRADE_ASSETS", value="DOGE,XRP", updated_at=now, updated_by="test"))
    await db_session.commit()
    
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = db_session
    
    with patch("polyflip.scheduler.jobs.ModelTrainer") as mock_trainer_cls, \
         patch("polyflip.scheduler.jobs.settings") as mock_settings, \
         patch("polyflip.scheduler.jobs.async_session", return_value=mock_session_cm):
        
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.train_model = AsyncMock(return_value=True)
        mock_settings.TRADE_ASSETS = "BTC,ETH"
        mock_settings.asset_list = ["BTC", "ETH", "DOGE", "XRP"]
        
        await retrain_job()
        
        called_assets = [call.args[0] for call in mock_trainer.train_model.call_args_list]
        assert "DOGE" in called_assets
        assert "XRP" in called_assets
        assert "BTC" not in called_assets
        assert "ETH" not in called_assets

@pytest.mark.asyncio
async def test_bug_05_06_group_shuffle_split(db_session):
    """
    Тест BUG-05 и BUG-06: Проверка, что ModelTrainer использует GroupShuffleSplit
    и корректно делит данные для калибровки без утечек групп.
    """
    snaps = []
    for i in range(100):
        # 10 рынков, по 10 снимков на каждый
        market_id = f"m_group_{i // 10}"
        snaps.append(MarketSnapshot(
            market_id=market_id, asset="BTC", time_left_min=float(i % 10 + 1),
            mid_price=0.8 if i % 2 == 0 else 0.2, spread=0.01,
            volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO",
            flip_vs_final=True if i % 2 == 0 else False,
            recorded_at=datetime.now(timezone.utc)
        ))
    db_session.add_all(snaps)
    await db_session.commit()
    
    trainer = ModelTrainer(db_session)
    
    # Проверим, что обучение отрабатывает успешно с GroupShuffleSplit
    from polyflip.models.trainer import settings as trainer_settings
    with patch.object(trainer_settings, "MIN_SAMPLES_FOR_MODEL", 10), \
         patch.object(trainer_settings, "ACTIVE_FEATURES", "mid_price,spread,time_left_min"), \
         patch("polyflip.models.trainer.GroupShuffleSplit", wraps=GroupShuffleSplit) as mock_gss:
        
        res = await trainer.train_model("BTC")
        assert res is True
        # Убедимся, что gss был вызван
        assert mock_gss.called

@pytest.mark.asyncio
async def test_bug_09_pnl_polymarket_fee(db_session):
    """
    Тест BUG-09: Расчет PnL в resolve_trades_job должен вычитать комиссию 0.2% Polymarket.
    """
    now = datetime.now(timezone.utc)
    
    win_trade = TradeHistory(
        market_id="m_win", asset="BTC", outcome_bought="YES",
        amount_usdc=10.0, executed_price=0.50, status="SUCCESS",
        predicted_flip_prob=0.0, active_features="",
        created_at=now - timedelta(minutes=30)
    )
    
    lose_trade = TradeHistory(
        market_id="m_lose", asset="BTC", outcome_bought="YES",
        amount_usdc=10.0, executed_price=0.50, status="SUCCESS",
        predicted_flip_prob=0.0, active_features="",
        created_at=now - timedelta(minutes=30)
    )
    
    db_session.add_all([win_trade, lose_trade])
    
    # Добавляем snapshots с результатами, указывая flip_vs_final=False для соблюдения NOT NULL
    db_session.add(MarketSnapshot(
        market_id="m_win", asset="BTC", mid_price=1.0, spread=0.0, time_left_min=0.0,
        volume_5min=0.0, price_velocity=0.0, hour_of_day=0, final_outcome="YES",
        flip_vs_final=False, recorded_at=now
    ))
    db_session.add(MarketSnapshot(
        market_id="m_lose", asset="BTC", mid_price=0.0, spread=0.0, time_left_min=0.0,
        volume_5min=0.0, price_velocity=0.0, hour_of_day=0, final_outcome="NO",
        flip_vs_final=False, recorded_at=now
    ))
    
    await db_session.commit()
    
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = db_session
    
    with patch("polyflip.scheduler.jobs.async_session", return_value=mock_session_cm):
        await resolve_trades_job()
    
    # Проверяем рассчитанный PnL
    q_win = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_win"))
    t_win = q_win.scalar_one()
    
    q_lose = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "m_lose"))
    t_lose = q_lose.scalar_one()
    
    # При выигрыше: валовая выплата = 10 / 0.5 = 20 USDC. С учетом 0.2% комиссии Polymarket: 20 * 0.998 = 19.96 USDC. PnL = 19.96 - 10.0 = 9.96.
    assert float(t_win.pnl) == 9.96
    # При проигрыше: позиция YES обесценивается до 0. Полный убыток равен размеру ставки (-10.0 USDC), комиссия Polymarket при этом не взимается.
    assert float(t_lose.pnl) == -10.00
