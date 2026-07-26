import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from polyflip.trading.engine import trade_worker_cycle, _ACTIVE_MARKETS
from polyflip.db.models import RuntimeSettings
from polyflip.collector.client import PolymarketClient
from polyflip.trading.trader import PolyTrader

class FakeMarket:
    def __init__(self, market_id: str, asset: str):
        self.market_id = market_id
        self.asset = asset
        self.end_time_est = datetime.now(timezone.utc)
        self.question = "Test market"

@pytest.mark.asyncio
async def test_engine_processes_market_exactly_once(db_session: AsyncSession):
    # Setup
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"))
    db_session.add(RuntimeSettings(key="TRADING_MODE", value="FAVORITE", updated_at=now, updated_by="test"))
    await db_session.commit()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()
    
    market = FakeMarket("test_market_1", "BTC")

    with patch("polyflip.trading.engine.load_eligible_markets", AsyncMock(return_value=[market])) as load_markets_mock, \
         patch("polyflip.trading.engine.check_market_guards", AsyncMock()) as guards_mock, \
         patch("polyflip.trading.engine.decide_favorite_mode", AsyncMock()) as decision_mock, \
         patch("polyflip.trading.engine.validate_pre_trade", AsyncMock()) as validator_mock, \
         patch("polyflip.trading.engine.execute_and_record", AsyncMock()) as execute_mock, \
         patch("polyflip.trading.engine.save_or_update_skipped_trade", AsyncMock()) as save_skipped_mock:
         
         guards_mock.return_value.passed = True
         guards_mock.return_value.existing_skipped = None
         
         decision_res = MagicMock()
         decision_res.decision_obj.action = "BUY_UP"
         decision_res.skip_reason = None
         decision_res.p_flip = 0.6
         decision_res.edge = 0.1
         decision_res.model_ver = 1
         decision_mock.return_value = decision_res
         
         validator_mock.return_value.valid = True
         
         await trade_worker_cycle(db_session, trader_mock, api_mock)
         
         # Verification
         assert guards_mock.call_count == 1, "Guards checked multiple times"
         assert decision_mock.call_count == 1, "Decision logic called multiple times"
         assert validator_mock.call_count == 1, "Validator called multiple times"
         assert execute_mock.call_count == 1, "Execute called multiple times"
         assert len(_ACTIVE_MARKETS) == 0, "Market was not removed from ACTIVE_MARKETS"


@pytest.mark.asyncio
async def test_engine_processes_multiple_markets_if_first_fails(db_session: AsyncSession):
    # Setup
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"))
    db_session.add(RuntimeSettings(key="TRADING_MODE", value="FAVORITE", updated_at=now, updated_by="test"))
    await db_session.commit()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()
    
    market1 = FakeMarket("test_market_FAIL", "BTC")
    market2 = FakeMarket("test_market_SUCCESS", "ETH")

    with patch("polyflip.trading.engine.load_eligible_markets", AsyncMock(return_value=[market1, market2])) as load_markets_mock, \
         patch("polyflip.trading.engine.check_market_guards", AsyncMock()) as guards_mock, \
         patch("polyflip.trading.engine.decide_favorite_mode", AsyncMock()) as decision_mock, \
         patch("polyflip.trading.engine.validate_pre_trade", AsyncMock()) as validator_mock, \
         patch("polyflip.trading.engine.execute_and_record", AsyncMock()) as execute_mock, \
         patch("polyflip.trading.engine.save_or_update_skipped_trade", AsyncMock()) as save_skipped_mock:
         
         # Let market1 fail at decision
         def guards_side_effect(db, m, *args, **kwargs):
             res = MagicMock()
             res.passed = True
             res.existing_skipped = None
             return res
             
         guards_mock.side_effect = guards_side_effect
         
         def decision_side_effect(m, *args, **kwargs):
             if m.market_id == "test_market_FAIL":
                 raise ValueError("Failed intentionally")
             res = MagicMock()
             res.decision_obj.action = "BUY_UP"
             res.skip_reason = None
             res.p_flip = 0.6
             res.edge = 0.1
             res.model_ver = 1
             return res
         
         decision_mock.side_effect = decision_side_effect
         validator_mock.return_value.valid = True
         
         await trade_worker_cycle(db_session, trader_mock, api_mock)
         
         # Verification
         assert guards_mock.call_count == 2
         assert execute_mock.call_count == 1
         assert execute_mock.call_args[0][2].market_id == "test_market_SUCCESS"
         assert len(_ACTIVE_MARKETS) == 0
