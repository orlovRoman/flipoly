import pytest
import pickle
from unittest.mock import patch, AsyncMock
from sqlalchemy import select
from datetime import datetime, timezone, timedelta
from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory, SlippageLog
from polyflip.trading.engine import trade_worker_cycle
from polyflip.api.slippage import get_slippage_summary, get_slippage_list

class MockModel:

    def __init__(self, proba):
        self.proba = proba
        self.feature_names_in_ = ['mid_price']

    def predict_proba(self, X):
        return [self.proba]

class DummyAsyncContextManager:

    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

def patch_session(db_session):
    return lambda: DummyAsyncContextManager(db_session)

@pytest.mark.asyncio
async def test_slippage_api_endpoints(db_session):
    import polyflip.api.slippage as slippage_module
    original_session = slippage_module.async_session
    slippage_module.async_session = patch_session(db_session)
    try:
        now = datetime.now(timezone.utc)
        db_session.add_all([SlippageLog(trade_id=1, market_id='m1', asset='BTC', outcome_bought='YES', expected_price=0.6, executed_price=0.62, slippage=0.02, slippage_pct=3.33, bet_size_usdc=10.0, slippage_cost_usdc=0.32, mode='PAPER', created_at=now), SlippageLog(trade_id=2, market_id='m2', asset='BTC', outcome_bought='NO', expected_price=0.4, executed_price=0.39, slippage=-0.01, slippage_pct=-2.5, bet_size_usdc=20.0, slippage_cost_usdc=-0.51, mode='PAPER', created_at=now + timedelta(seconds=1)), SlippageLog(trade_id=3, market_id='m3', asset='ETH', outcome_bought='YES', expected_price=0.7, executed_price=0.71, slippage=0.01, slippage_pct=1.43, bet_size_usdc=10.0, slippage_cost_usdc=0.14, mode='PAPER', created_at=now + timedelta(seconds=2))])
        await db_session.commit()
        lst = await get_slippage_list(limit=2)
        assert len(lst) == 2
        assert lst[0].trade_id == 3
        assert lst[1].trade_id == 2
        summary = await get_slippage_summary()
        assert len(summary) == 2
        btc = [s for s in summary if s['asset'] == 'BTC'][0]
        assert btc['count'] == 2
        assert round(btc['avg_slippage'], 3) == 0.005
        assert round(btc['total_cost_usdc'], 2) == -0.19
        eth = [s for s in summary if s['asset'] == 'ETH'][0]
        assert eth['count'] == 1
        assert eth['total_cost_usdc'] == 0.14
    finally:
        slippage_module.async_session = original_session