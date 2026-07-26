import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from polyflip.db.models import TradeHistory, LiveMarket, RuntimeSettings
from polyflip.trading.position_closer import execute_position_exit
from polyflip.trading.stoploss_worker import stoploss_worker_cycle
from polyflip.trading.recovery_worker import recovery_worker_cycle

@pytest_asyncio.fixture
async def prep_exit_failed_trade(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    
    db_session.add(RuntimeSettings(key="STOP_LOSS_ENABLED", value="true", updated_at=now, updated_by="test"))
    db_session.add(RuntimeSettings(key="POLYMARKET_FEE_RATE", value="0.0", updated_at=now, updated_by="test"))
    
    trade = TradeHistory(
        market_id="fail_m1",
        asset="ETH",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.50,
        predicted_flip_prob=0.8,
        active_features="",
        status="SUCCESS",
        position_status="OPEN", 
        stop_loss_status="ACTIVE",
        stop_loss_price=0.40,
        stop_loss_pct=20.0,
        exit_attempts=0,
        mode="PAPER",
        created_at=now
    )
    db_session.add(trade)
    
    market = LiveMarket(
        market_id="fail_m1",
        asset="ETH",
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
async def test_step8_exit_failed_and_retry(db_session: AsyncSession, prep_exit_failed_trade):
    """
    Проверить сценарий фейла при продаже (8.2, 8.3, 8.4):
    1. position_closer получает ошибку от PolyTrader -> position_status = EXIT_FAILED
    2. Воркер (stoploss) подхватывает EXIT_FAILED и пробует снова (увеличивая exit_attempts)
    """
    trade = prep_exit_failed_trade
    
    # Мокаем трейдера, чтобы он бросал ошибку (или возвращал FAILED)
    from tests.helpers import make_dummy_execution
    trader_mock = AsyncMock()
    trader_mock.execute_trade = AsyncMock(return_value=make_dummy_execution(
        mode="PAPER", status="REJECTED", error_msg="Timeout"
    ))
    
    api_mock = AsyncMock()
    api_mock.get_market_prices = AsyncMock(return_value={"best_bid": 0.35}) # trigger stoploss
    
    import polyflip.trading.stoploss_worker as slw
    orig_sl = slw.evaluate_stop_loss
    
    from polyflip.trading.stoploss import StopLossDecision
    slw.evaluate_stop_loss = MagicMock(return_value=StopLossDecision(should_sell=True, stop_price=0.40, current_price=0.35, reason="test"))
    
    try:
        # Цикл 1: Попытка закрыть позицию -> ошибка
        await stoploss_worker_cycle(db_session, trader_mock, api_mock)
        await db_session.refresh(trade)
        
        assert trade.position_status == "EXIT_FAILED"
        assert trade.exit_attempts == 1
        assert trader_mock.execute_trade.call_count == 1
        
        # Цикл 2: Воркер (recovery) должен подхватить позицию, т.к. она EXIT_FAILED
        trader_mock.execute_trade = AsyncMock(return_value=make_dummy_execution(
            mode="PAPER", status="FILLED", executed_price=0.35, executed_usdc=7.0
        ))
        
        balance_mock = MagicMock()
        balance_mock.status = "PAPER"
        balance_mock.available_shares = 20.0
        trader_mock.get_balance = AsyncMock(return_value=balance_mock)
        
        # Advance time so recovery worker considers it stale
        from datetime import timedelta
        trade.exit_claimed_at = trade.exit_claimed_at - timedelta(minutes=10)
        await db_session.commit()
        
        await recovery_worker_cycle(db_session, trader_mock, api_mock)
        await db_session.refresh(trade)
        
        assert trade.position_status == "CLOSED"
        assert trade.exit_attempts == 2
        assert trader_mock.execute_trade.call_count == 1  # сбросился мок
        
    finally:
        slw.evaluate_stop_loss = orig_sl
