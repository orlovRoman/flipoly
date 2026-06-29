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
        self.feature_names_in_ = ["mid_price"]
    def predict_proba(self, X):
        return [self.proba]

# Патчим async_session в slippage.py для использования тестовой db_session
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
async def test_slippage_logged_after_successful_trade(db_session):
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
        RuntimeSettings(key="AUTO_DEAD_ZONE", value="false", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_EDGE", value="0.40", updated_at=now, updated_by="test"),
        RuntimeSettings(key="KELLY_ENABLED", value="false", updated_at=now, updated_by="test"),
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
         # Buy price was 0.61 (best_ask), but executed_price is 0.63 -> slippage = +0.02
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "executed_usdc": 10.0, "executed_price": 0.63, "mode": "PAPER"})
         
         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.61})
         
         await trade_worker_cycle(db_session, mock_trader, mock_api)
         
         # Check TradeHistory
         trades = (await db_session.execute(select(TradeHistory))).scalars().all()
         assert len(trades) == 1
         trade = trades[0]
         
         # Check SlippageLog
         slippages = (await db_session.execute(select(SlippageLog))).scalars().all()
         assert len(slippages) == 1
         slip = slippages[0]
         assert slip.trade_id == trade.id
         assert slip.market_id == "m1"
         assert slip.asset == "BTC"
         assert slip.outcome_bought == "YES"
         assert round(slip.expected_price, 2) == 0.61
         assert round(slip.executed_price, 2) == 0.63
         assert round(slip.slippage, 2) == 0.02
         assert round(slip.slippage_pct, 2) == 3.28  # 0.02 / 0.61 * 100
         assert slip.mode == "PAPER"
         expected_cost = round(0.02 * (10.0 / 0.63), 4)  # ≈ 0.3175
         assert abs(slip.slippage_cost_usdc - expected_cost) < 0.001

@pytest.mark.asyncio
async def test_slippage_api_endpoints(db_session):
    import polyflip.api.slippage as slippage_module
    original_session = slippage_module.async_session
    slippage_module.async_session = patch_session(db_session)
    
    try:
        now = datetime.now(timezone.utc)
        # Add mock logs
        db_session.add_all([
            SlippageLog(
                trade_id=1, market_id="m1", asset="BTC", outcome_bought="YES",
                expected_price=0.60, executed_price=0.62, slippage=0.02, slippage_pct=3.33,
                bet_size_usdc=10.0, slippage_cost_usdc=0.32, mode="PAPER", created_at=now
            ),
            SlippageLog(
                trade_id=2, market_id="m2", asset="BTC", outcome_bought="NO",
                expected_price=0.40, executed_price=0.39, slippage=-0.01, slippage_pct=-2.5,
                bet_size_usdc=20.0, slippage_cost_usdc=-0.51, mode="PAPER", created_at=now + timedelta(seconds=1)
            ),
            SlippageLog(
                trade_id=3, market_id="m3", asset="ETH", outcome_bought="YES",
                expected_price=0.70, executed_price=0.71, slippage=0.01, slippage_pct=1.43,
                bet_size_usdc=10.0, slippage_cost_usdc=0.14, mode="PAPER", created_at=now + timedelta(seconds=2)
            ),
        ])
        await db_session.commit()
        
        # Test get_slippage_list
        lst = await get_slippage_list(limit=2)
        assert len(lst) == 2
        assert lst[0].trade_id == 3  # ordered by desc created_at
        assert lst[1].trade_id == 2
        
        # Test get_slippage_summary
        summary = await get_slippage_summary()
        assert len(summary) == 2  # BTC, ETH
        
        btc = [s for s in summary if s["asset"] == "BTC"][0]
        assert btc["count"] == 2
        assert round(btc["avg_slippage"], 3) == 0.005  # (0.02 - 0.01) / 2
        assert round(btc["total_cost_usdc"], 2) == -0.19  # 0.32 - 0.51
        
        eth = [s for s in summary if s["asset"] == "ETH"][0]
        assert eth["count"] == 1
        assert eth["total_cost_usdc"] == 0.14
        
    finally:
        slippage_module.async_session = original_session
