import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, SlippageLog
from polyflip.trading.stoploss_worker import stoploss_worker_cycle


@pytest.mark.asyncio
async def test_worker_skips_when_disabled(db_session):
    """Воркер ничего не делает, если STOP_LOSS_ENABLED != true."""
    db_session.add(RuntimeSettings(
        key="STOP_LOSS_ENABLED", 
        value="false",
        updated_at=datetime.now(timezone.utc),
        updated_by="test"
    ))
    await db_session.flush()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()

    await stoploss_worker_cycle(db_session, trader_mock, api_mock)

    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_worker_expired_by_market_end_time(db_session):
    """Если now >= market_end_time → статус EXPIRED, sell не вызывается."""
    db_session.add(RuntimeSettings(
        key="STOP_LOSS_ENABLED", 
        value="true",
        updated_at=datetime.now(timezone.utc),
        updated_by="test"
    ))
    
    now = datetime.now(timezone.utc)
    past_end = now - timedelta(minutes=5)
    
    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.3,
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_price=0.25,
        market_end_time=past_end,
        created_at=now,
        active_features="test"
    )
    db_session.add(trade)
    await db_session.flush()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()

    await stoploss_worker_cycle(db_session, trader_mock, api_mock)

    await db_session.refresh(trade)
    assert trade.stop_loss_status == "EXPIRED"
    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_worker_expired_when_market_missing_from_live(db_session):
    """Если рынок отсутствует в LiveMarket → EXPIRED."""
    db_session.add(RuntimeSettings(
        key="STOP_LOSS_ENABLED", 
        value="true",
        updated_at=datetime.now(timezone.utc),
        updated_by="test"
    ))
    
    now = datetime.now(timezone.utc)
    future_end = now + timedelta(minutes=10)
    
    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.3,
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_price=0.25,
        market_end_time=future_end,
        created_at=now,
        active_features="test"
    )
    db_session.add(trade)
    await db_session.flush()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()

    await stoploss_worker_cycle(db_session, trader_mock, api_mock)

    await db_session.refresh(trade)
    assert trade.stop_loss_status == "EXPIRED"
    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_worker_no_trigger_when_bid_above_stop(db_session):
    """При bid > stop_price sell не выполняется."""
    db_session.add(RuntimeSettings(
        key="STOP_LOSS_ENABLED", 
        value="true",
        updated_at=datetime.now(timezone.utc),
        updated_by="test"
    ))
    
    now = datetime.now(timezone.utc)
    future_end = now + timedelta(minutes=10)
    
    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.3,
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_pct=50.0,
        stop_loss_price=0.25,
        market_end_time=future_end,
        created_at=now,
        active_features="test"
    )
    market = LiveMarket(
        market_id="m1",
        asset="BTC",
        question="BTC up?",
        yes_token_id="yes1",
        no_token_id="no1",
        end_time_est=future_end,
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        volume_5min=100.0,
        price_velocity=0.0,
        last_updated=now
    )
    db_session.add_all([trade, market])
    await db_session.flush()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()
    api_mock.get_market_prices.return_value = {"best_bid": 0.30}

    await stoploss_worker_cycle(db_session, trader_mock, api_mock)

    await db_session.refresh(trade)
    assert trade.stop_loss_status == "ACTIVE"
    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_worker_triggers_sell_when_bid_below_stop(db_session):
    """При bid <= stop_price выполняется sell-ордер и статус меняется на TRIGGERED."""
    db_session.add(RuntimeSettings(
        key="STOP_LOSS_ENABLED", 
        value="true",
        updated_at=datetime.now(timezone.utc),
        updated_by="test"
    ))
    
    now = datetime.now(timezone.utc)
    future_end = now + timedelta(minutes=10)
    
    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.3,
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_pct=50.0,
        stop_loss_price=0.25,
        market_end_time=future_end,
        created_at=now,
        active_features="test"
    )
    market = LiveMarket(
        market_id="m1",
        asset="BTC",
        question="BTC up?",
        yes_token_id="yes1",
        no_token_id="no1",
        end_time_est=future_end,
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        volume_5min=100.0,
        price_velocity=0.0,
        last_updated=now
    )
    db_session.add_all([trade, market])
    await db_session.flush()

    trader_mock = AsyncMock()
    trader_mock.execute_trade.return_value = {
        "status": "SUCCESS",
        "mode": "PAPER",
        "executed_price": 0.20
    }
    
    api_mock = AsyncMock()
    api_mock.get_market_prices.return_value = {"best_bid": 0.20}

    await stoploss_worker_cycle(db_session, trader_mock, api_mock)

    await db_session.refresh(trade)
    assert trade.stop_loss_status == "TRIGGERED"
    trader_mock.execute_trade.assert_called_once_with(
        market_id="m1",
        token_id="yes1",
        side="SELL",
        price=0.20,
        size=20.0
    )
    assert trade.pnl == pytest.approx(-6.008, abs=1e-3)
    assert trade.stop_loss_sell_price == 0.20

    # Проверим, что SlippageLog был создан
    res = await db_session.execute(select(SlippageLog).where(SlippageLog.trade_id == trade.id))
    slip_log = res.scalar_one_or_none()
    assert slip_log is not None
    assert slip_log.expected_price == 0.20
    assert slip_log.executed_price == 0.20
    assert slip_log.slippage == 0.0


@pytest.mark.asyncio
async def test_worker_skips_trade_with_missing_stop_loss_pct(db_session):
    """Позиция с stop_loss_pct=None не должна падать — статус EXPIRED."""
    db_session.add(RuntimeSettings(
        key="STOP_LOSS_ENABLED", value="true",
        updated_at=datetime.now(timezone.utc), updated_by="test"
    ))
    now = datetime.now(timezone.utc)
    trade = TradeHistory(
        market_id="m_none_pct", asset="BTC", outcome_bought="YES",
        amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.3,
        status="SUCCESS", stop_loss_status="ACTIVE",
        stop_loss_pct=None,       # ← ключевое
        stop_loss_price=0.25,
        market_end_time=now + timedelta(minutes=10),
        created_at=now, active_features="test"
    )
    db_session.add(trade)
    await db_session.flush()

    market = LiveMarket(
        market_id="m_none_pct", asset="BTC", question="?",
        yes_token_id="y1", no_token_id="n1",
        end_time_est=now + timedelta(minutes=10),
        current_yes_price=0.5, current_no_price=0.5,
        current_spread=0.01, volume_5min=0.0,
        price_velocity=0.0, last_updated=now
    )
    db_session.add(market)
    await db_session.flush()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()
    api_mock.get_market_prices.return_value = {"best_bid": 0.20}

    await stoploss_worker_cycle(db_session, trader_mock, api_mock)

    await db_session.refresh(trade)
    assert trade.stop_loss_status == "EXPIRED"
    trader_mock.execute_trade.assert_not_called()
