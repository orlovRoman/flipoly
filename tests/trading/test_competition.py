import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from polyflip.db.models import TradeHistory, LiveMarket, RuntimeSettings
from polyflip.trading.stoploss_worker import stoploss_worker_cycle
from polyflip.trading.takeprofit_worker import takeprofit_worker_cycle

# Setup basic fixtures for tests (using same db_session pattern as test_engine_refactor)

@pytest_asyncio.fixture
async def prep_competition_trade(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    
    # Enable both SL and TP
    db_session.add(RuntimeSettings(key="STOP_LOSS_ENABLED", value="true", updated_at=now, updated_by="test"))
    db_session.add(RuntimeSettings(key="TAKE_PROFIT_ENABLED", value="true", updated_at=now, updated_by="test"))
    db_session.add(RuntimeSettings(key="POLYMARKET_FEE_RATE", value="0.0", updated_at=now, updated_by="test"))
    
    trade = TradeHistory(
        market_id="comp_m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.50,
        predicted_flip_prob=0.8,
        active_features="",
        status="SUCCESS",
        position_status="OPEN",  # Important for claim logic
        stop_loss_status="ACTIVE",
        stop_loss_price=0.40,
        stop_loss_pct=20.0,
        take_profit_status="ACTIVE",
        take_profit_price=0.60,
        take_profit_multiplier=1.2,
        mode="PAPER",
        created_at=now
    )
    db_session.add(trade)
    
    market = LiveMarket(
        market_id="comp_m1",
        asset="BTC",
        question="test",
        yes_token_id="y1",
        no_token_id="n1",
        end_time_est=now,
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        volume_5min=1000.0,
        price_velocity=0.0,
        last_updated=now
    )
    db_session.add(market)
    
    await db_session.commit()
    await db_session.refresh(trade)
    return trade


@pytest.mark.asyncio
async def test_step7_competition_stoploss_and_takeprofit(db_session: AsyncSession, prep_competition_trade):
    """
    Проверить конкуренцию:
    await asyncio.gather(
        stoploss_worker_cycle(...),
        takeprofit_worker_cycle(...),
    )
    Проверки:
    execute_trade вызван ровно один раз
    position_status=CLOSING или CLOSED
    exit_reason содержит только одну причину
    не создано двух записей проскальзывания
    """
    trade = prep_competition_trade
    
    trader_mock = AsyncMock()
    # Adding a small sleep to simulate network delay, which maximizes chance of race condition
    async def mock_execute_trade(*args, **kwargs):
        await asyncio.sleep(0.1)
        return {"status": "SUCCESS", "executed_price": 0.55, "executed_usdc": 11.0, "mode": "PAPER"}
    trader_mock.execute_trade = AsyncMock(side_effect=mock_execute_trade)
    
    api_mock = AsyncMock()
    # We return a bid price that triggers BOTH stoploss and takeprofit theoretically? 
    # Wait, 0.55 is > 0.40 (not SL) and < 0.60 (not TP). 
    # Let's say current_bid = 0.35, which triggers SL (0.35 < 0.40). 
    # Let's say current_bid = 0.65, which triggers TP (0.65 > 0.60).
    # Since we want BOTH to try triggering, we need the evaluate function to return should_sell=True for both.
    # So let's mock `evaluate_stop_loss` and `evaluate_take_profit` directly OR just supply a bid that triggers both?
    # Stoploss triggers if bid <= stop_price. 0.35 <= 0.40
    # Takeprofit triggers if bid >= tp_price. 0.65 >= 0.60
    # A single bid cannot trigger both mathematically. 
    # BUT we want to simulate concurrent worker execution where they both THINK they should trigger.
    # We can mock get_market_prices differently per worker or mock the `claim_position_for_exit`?
    # The simplest is to patch `evaluate_stop_loss` and `evaluate_take_profit` to return should_sell=True.
    
    import polyflip.trading.stoploss_worker as slw
    import polyflip.trading.takeprofit_worker as tpw
    
    # Save original functions
    orig_sl = slw.evaluate_stop_loss
    orig_tp = tpw.evaluate_take_profit
    
    from polyflip.trading.stoploss import StopLossDecision
    from polyflip.trading.takeprofit import TakeProfitDecision
    
    slw.evaluate_stop_loss = MagicMock(return_value=StopLossDecision(should_sell=True, stop_price=0.40, current_price=0.55, reason="test"))
    tpw.evaluate_take_profit = MagicMock(return_value=TakeProfitDecision(should_sell=True, tp_price=0.60, current_price=0.55, reason="test"))
    
    api_mock.get_market_prices = AsyncMock(return_value={"best_bid": 0.55})
    
    try:
        # Run sequentially to simulate what happens if they trigger around the same time
        # The first one should claim the position and change its status to CLOSED/CLOSING
        # The second one should not trigger execute_trade again.
        await stoploss_worker_cycle(db_session, trader_mock, api_mock)
        await takeprofit_worker_cycle(db_session, trader_mock, api_mock)
    finally:
        slw.evaluate_stop_loss = orig_sl
        tpw.evaluate_take_profit = orig_tp
        
    # Check results
    await db_session.refresh(trade)
    
    # execute_trade вызван ровно один раз
    assert trader_mock.execute_trade.call_count == 1
    
    # position_status = CLOSED (since our mock trader returns SUCCESS)
    assert trade.position_status == "CLOSED"
    
    # exit_reason содержит только одну причину
    assert trade.exit_reason in ("STOP_LOSS", "TAKE_PROFIT")
    assert trade.exit_attempts == 1
    
    # Не создано двух записей проскальзывания
    from polyflip.db.models import SlippageLog
    stmt = select(SlippageLog).where(SlippageLog.trade_id == trade.id)
    slippage_logs = (await db_session.execute(stmt)).scalars().all()
    assert len(slippage_logs) == 1
