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
        self.feature_names_in_ = ['mid_price']

    def predict_proba(self, X):
        return [self.proba]

@pytest.mark.asyncio
async def test_engine_skips_when_trading_disabled(db_session):
    api_client_mock = AsyncMock()
    await trade_worker_cycle(db_session, api_client_mock)
    res = await db_session.execute(select(TradeHistory))
    assert len(res.scalars().all()) == 0

@pytest.mark.asyncio
async def test_save_or_update_no_extra_select(db_session):
    """Функция save_or_update_skipped_trade не должна делать SELECT, если передан existing_skipped."""
    from polyflip.trading.engine import save_or_update_skipped_trade
    original_execute = db_session.execute
    mock_execute = AsyncMock(side_effect=original_execute)
    db_session.execute = mock_execute

    class FakeMarket:
        market_id = 'm1'
        asset = 'BTC'
    now = datetime.now(timezone.utc)
    existing = TradeHistory(market_id='m1', asset='BTC', outcome_bought='NONE', amount_usdc=0.0, remaining_shares=20.0, executed_price=0.0, predicted_flip_prob=0.5, active_features='', model_version=1, status='SKIPPED', error_msg='Old reason', created_at=now)
    await save_or_update_skipped_trade(db_session=db_session, market=FakeMarket(), reason='New reason', p_flip_val=0.6, model_version=1, start_time=now, existing_skipped=existing)
    assert mock_execute.call_count == 0
    assert existing.error_msg == 'New reason'
    assert existing.predicted_flip_prob == 0.6

@pytest.mark.asyncio
async def test_engine_skips_when_no_fresh_prices(db_session):
    """При отсутствии цен от API движок должен записывать пропуск (SKIPPED)."""
    now = datetime.now(timezone.utc)
    settings = [RuntimeSettings(key='TRADING_ENABLED', value='true', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_EXECUTION_TIME_SEC', value='30', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_BET_SIZE_USDC', value='10.0', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_NO_FLIP_THRESHOLD', value='0.15', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_FLIP_THRESHOLD', value='0.85', updated_at=now, updated_by='test'), RuntimeSettings(key='ACTIVE_FEATURES', value='mid_price', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_MIN_PRICE', value='0.05', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_MAX_PRICE', value='0.95', updated_at=now, updated_by='test')]
    db_session.add_all(settings)
    market = LiveMarket(market_id='m_no_price', asset='BTC', question='Test?', current_yes_price=0.6, current_no_price=0.4, current_spread=0.01, volume_5min=100.0, price_velocity=0.0, end_time_est=now + timedelta(seconds=300), yes_token_id='t_yes', no_token_id='t_no', last_updated=now)
    db_session.add(market)
    model = MockModel([0.95, 0.05])
    db_session.add(ModelRegistry(asset='BTC', model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features='mid_price', trained_at=now))
    await db_session.commit()
    with patch('polyflip.trading.engine.PolymarketClient') as mock_api_cls:
        mock_api = mock_api_cls.return_value
        mock_api.get_market_prices = AsyncMock(return_value={})
        mock_api.close = AsyncMock()
        await trade_worker_cycle(db_session, mock_api)
        res = await db_session.execute(select(TradeHistory))
        trades = res.scalars().all()
        assert len(trades) == 1
        assert trades[0].status == 'SKIPPED'
        assert 'Failed to fetch fresh Polymarket YES price' in trades[0].error_msg

@pytest.mark.asyncio
async def test_engine_skips_when_clob_error(db_session):
    now = datetime.now(timezone.utc)
    settings = [RuntimeSettings(key='TRADING_ENABLED', value='true', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_EXECUTION_TIME_SEC', value='30', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_BET_SIZE_USDC', value='10.0', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_NO_FLIP_THRESHOLD', value='0.15', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_FLIP_THRESHOLD', value='0.85', updated_at=now, updated_by='test'), RuntimeSettings(key='ACTIVE_FEATURES', value='mid_price', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_MIN_PRICE', value='0.05', updated_at=now, updated_by='test'), RuntimeSettings(key='TRADE_MAX_PRICE', value='0.95', updated_at=now, updated_by='test'), RuntimeSettings(key='DEAD_ZONE_WIDTH', value='0.02', updated_at=now, updated_by='test'), RuntimeSettings(key='OUTS_MIN_EDGE', value='0.05', updated_at=now, updated_by='test')]
    db_session.add_all(settings)
    market = LiveMarket(market_id='m_clob_err', asset='BTC', question='Test?', current_yes_price=0.6, current_no_price=0.4, current_spread=0.01, volume_5min=100.0, price_velocity=0.0, end_time_est=now + timedelta(seconds=300), yes_token_id='t_yes', no_token_id='t_no', last_updated=now)
    db_session.add(market)
    model = MockModel([0.95, 0.05])
    db_session.add(ModelRegistry(asset='BTC', model_blob=pickle.dumps(model), is_active=True, version=1, accuracy=0.9, features='mid_price', trained_at=now))
    await db_session.commit()
    with patch('polyflip.trading.engine.PolymarketClient') as mock_api_cls:
        mock_api = mock_api_cls.return_value
        mock_api.get_market_prices = AsyncMock(return_value={'error': 'API HTTP Error 429'})
        mock_api.close = AsyncMock()
        await trade_worker_cycle(db_session, mock_api)
        res = await db_session.execute(select(TradeHistory))
        trades = res.scalars().all()
        target_trade = next((t for t in trades if t.market_id == 'm_clob_err'))
        assert target_trade.status == 'SKIPPED'
        assert 'Failed to fetch fresh Polymarket YES price' in target_trade.error_msg