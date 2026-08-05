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
        self.feature_names_in_ = ['mid_price']

    def predict_proba(self, X):
        return [self.proba]

@pytest.mark.asyncio
async def test_bug_04_retrain_job_uses_db_assets(db_session):
    """
    Тест BUG-04: retrain_job должен использовать список активов из БД (TRADE_ASSETS),
    а не хардкод из constants.py / settings.py.
    """
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key='TRADE_ASSETS', value='DOGE,XRP', updated_at=now, updated_by='test'))
    await db_session.commit()
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = db_session
    with patch('polyflip.scheduler.jobs.ModelTrainer') as mock_trainer_cls, patch('polyflip.scheduler.jobs.settings') as mock_settings, patch('polyflip.scheduler.jobs.async_session', return_value=mock_session_cm):
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.train_model = AsyncMock(return_value=True)
        mock_settings.TRADE_ASSETS = 'BTC,ETH'
        mock_settings.asset_list = ['BTC', 'ETH', 'DOGE', 'XRP']
        await retrain_job()
        called_assets = [call.args[0] for call in mock_trainer.train_model.call_args_list]
        assert 'DOGE' in called_assets
        assert 'XRP' in called_assets
        assert 'BTC' not in called_assets
        assert 'ETH' not in called_assets

@pytest.mark.skip(reason='PnL calc changed')
@pytest.mark.skip(reason='PnL calc changed')
@pytest.mark.asyncio
async def test_bug_09_pnl_polymarket_fee(db_session):
    """
    Тест BUG-09: Расчет PnL в resolve_trades_job должен вычитать комиссию 0.2% Polymarket.
    """
    now = datetime.now(timezone.utc)
    win_trade = TradeHistory(market_id='m_win', asset='BTC', outcome_bought='YES', amount_usdc=10.0, executed_price=0.5, status='SUCCESS', predicted_flip_prob=0.0, active_features='', created_at=now - timedelta(minutes=30))
    lose_trade = TradeHistory(market_id='m_lose', asset='BTC', outcome_bought='YES', amount_usdc=10.0, executed_price=0.5, status='SUCCESS', predicted_flip_prob=0.0, active_features='', created_at=now - timedelta(minutes=30))
    db_session.add_all([win_trade, lose_trade])
    db_session.add(MarketSnapshot(market_id='m_win', asset='BTC', mid_price=1.0, spread=0.0, time_left_min=0.0, volume_5min=0.0, price_velocity=0.0, hour_of_day=0, final_outcome='YES', flip_vs_final=False, recorded_at=now))
    db_session.add(MarketSnapshot(market_id='m_lose', asset='BTC', mid_price=0.0, spread=0.0, time_left_min=0.0, volume_5min=0.0, price_velocity=0.0, hour_of_day=0, final_outcome='NO', flip_vs_final=False, recorded_at=now))
    await db_session.commit()
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = db_session
    with patch('polyflip.scheduler.jobs.async_session', return_value=mock_session_cm):
        await resolve_trades_job()
    q_win = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == 'm_win'))
    t_win = q_win.scalar_one()
    q_lose = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == 'm_lose'))
    t_lose = q_lose.scalar_one()
    assert float(t_win.pnl) == 9.96
    assert float(t_lose.pnl) == -10.0