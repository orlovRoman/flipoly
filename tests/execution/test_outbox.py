import pytest
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import select

from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.db.execution_models import ExecutionRequest
from polyflip.trading.trade_recorder import execute_and_record
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.pre_trade_validator import PreTradeValidation
from polyflip.execution.config import ExecutionSettings, ExecutionMode

@pytest.mark.asyncio
async def test_trade_recorder_creates_execution_request(db_session):
    # Setup
    market = LiveMarket(
        market_id="test_market_outbox",
        asset="BTC",
        question="BTC > 100k?",
        yes_token_id="yes_tok",
        no_token_id="no_tok",
        end_time_est=datetime.now(timezone.utc),
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        last_updated=datetime.now(timezone.utc)
    )
    db_session.add(market)
    await db_session.commit()

    decision = TradeDecision(action="BUY_YES", p_up=0.6, strike=0.5, strategy_type="ML", buy_price=0.5, bet_size_usdc=10.0, reason="test")
    validation = PreTradeValidation(valid=True, buy_price=0.5, actual_bet_size=10.0, edge=0.1, skip_reason=None)
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.stop_loss_enabled = False
    cfg.take_profit_enabled = False

    import os
    os.environ["EXECUTION_MODE"] = "PAPER"

    # Execution
    await execute_and_record(
        db_session=db_session,
        market=market,
        decision_obj=decision,
        validation=validation,
        asset_mode="ML",
        active_features="ML",
        p_flip=0.6,
        model_ver=1,
        cfg=cfg,
        existing_skipped=None,
        start_time=datetime.now(timezone.utc)
    )

    # Verification
    # Check that TradeHistory is created
    result_trade = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "test_market_outbox"))
    trade = result_trade.scalar_one_or_none()
    assert trade is not None
    assert trade.position_status == "OPENING"
    
    # Check that ExecutionRequest is created
    result_req = await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.market_id == "test_market_outbox"))
    req = result_req.scalar_one_or_none()
    
    assert req is not None
    assert req.intent == "OPEN"
    assert req.state == "READY"
    assert req.target_amount_usdc == Decimal("10.0")
    assert req.requested_shares == Decimal("20.0") # 10 / 0.5 = 20.0
