import pytest
import datetime
from decimal import Decimal
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket
from polyflip.db.execution_models import ExecutionRequest, ExposureReservation, ExecutionEvent
from polyflip.execution.risk_checks import check_risk_limits
from polyflip.execution.config import ExecutionMode
from polyflip.api.execution_api import KillSwitchRequest
from polyflip.trading.pre_trade_validator import validate_pre_trade
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision

@pytest.mark.asyncio
async def test_daily_loss_limit(db_session: AsyncSession):
    now = datetime.datetime.now(datetime.timezone.utc)
    # Set limit to -100
    db_session.add(RuntimeSettings(key="DAILY_LOSS_LIMIT_USDC", value="-100", updated_at=now, updated_by="test"))
    # Add a trade with 0 PnL
    trade = TradeHistory(
        market_id="m1", asset="a", outcome_bought="Yes", executed_price=0.5,
        strategy_type="LIGHTGBM", predicted_flip_prob=0.8, market_role="FAVORITE",
        position_status="CLOSED", status="SUCCESS", amount_usdc=10.0,
        position_accounting_version=1, position_version=1, active_features="X",
        mode="PAPER", entry_filled_shares=20.0, entry_cost_usdc=10.0,
        remaining_shares=0.0, realized_pnl_usdc=0.0, created_at=now, closed_at=now
    )
    db_session.add(trade)
    await db_session.commit()
    
    error = await check_risk_limits(db_session, "OPEN", Decimal("10"), "PAPER")
    assert error is None, "PnL 0 should not be blocked by -100 limit"

class MockClient:
    async def get_market_prices(self, token_id):
        return {"best_ask": "0.5", "best_bid": "0.4"}

@pytest.mark.asyncio
async def test_released_reservation_not_counted_in_pretrade(db_session: AsyncSession):
    now = datetime.datetime.now(datetime.timezone.utc)
    market = LiveMarket(market_id="m1", asset="a", yes_token_id="y1", no_token_id="n1",
                        question="q")
    # Set exposure limit to 50
    db_session.add(RuntimeSettings(key="MAX_TOTAL_EXPOSURE_USDC", value="50", updated_at=now, updated_by="test"))
    
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.bet_size = 10.0
    cfg.flip_threshold = 0.5
    cfg.max_price_drift = 0.05
    cfg.bypass_bet_size_check = False
    cfg.max_exposure_pct = 100.0
    cfg.favorite_min_edge = 0.05
    cfg.fee_rate = 0.0
    cfg.slippage_rate = 0.0
    cfg.trade_min_price = 0.01
    cfg.trade_max_price = 0.99
    cfg.max_bet_size_usdc = 20.0
    cfg.max_bet_edge = 0.40
    cfg.capital = 100.0
    cfg.bet_sizing_mode = "scaled"
    decision = TradeDecision(action="BUY_YES", buy_price=0.5, bet_size_usdc=10.0,
                             strategy_type="LIGHTGBM_TREND", edge=0.1, p_win_effective=0.6, reason="test")
    
    trade = TradeHistory(
        market_id="m1", asset="a", outcome_bought="Yes", executed_price=0.5,
        strategy_type="LIGHTGBM", predicted_flip_prob=0.8, market_role="FAVORITE",
        position_status="CLOSED", status="SUCCESS", amount_usdc=10.0,
        position_accounting_version=1, position_version=1, active_features="X",
        mode="PAPER", entry_filled_shares=20.0, entry_cost_usdc=10.0,
        remaining_shares=0.0, realized_pnl_usdc=0.0, created_at=now, closed_at=now
    )
    db_session.add(trade)
    await db_session.flush()

    res = ExposureReservation(
        id=uuid4(), request_id=uuid4(), trade_history_id=trade.id, market_id="m1", amount_usdc=Decimal("50"),
        expires_at=now + datetime.timedelta(hours=1), released_at=now
    )
    db_session.add(res)
    await db_session.commit()
    
    # 50 is released, so requesting 10 should be valid (limit is 50 per market)
    val = await validate_pre_trade(db_session, MockClient(), market, decision, cfg, "LIGHTGBM", 0.05, 0.99, 0.6, 1)
    assert val.valid is True, f"Blocked by reservation that is released: {val.skip_reason}"

def test_killswitch_request_schema():
    req = KillSwitchRequest(enabled=False)
    assert req.enabled is False
