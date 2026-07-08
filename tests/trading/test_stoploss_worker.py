import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket
from polyflip.trading.stoploss_worker import stoploss_worker_cycle


@pytest.mark.asyncio
async def test_worker_skips_when_disabled():
    """Воркер ничего не делает, если STOP_LOSS_ENABLED != true."""
    db_mock = AsyncMock()
    # Mock Settings: STOP_LOSS_ENABLED = "false"
    mock_setting = MagicMock(key="STOP_LOSS_ENABLED", value="false")
    db_mock.execute.return_value.scalar_one_or_none.return_value = mock_setting

    trader_mock = MagicMock()
    api_mock = MagicMock()

    await stoploss_worker_cycle(db_mock, trader_mock, api_mock)

    # Должен быть только один запрос к БД (выборка настроек)
    assert db_mock.execute.call_count == 1
    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_worker_expired_by_market_end_time():
    """Если now >= market_end_time → статус EXPIRED, sell не вызывается."""
    db_mock = AsyncMock()
    # Mock Settings: STOP_LOSS_ENABLED = "true"
    mock_setting = MagicMock(key="STOP_LOSS_ENABLED", value="true")
    
    # Mock Trade: ACTIVE, но market_end_time в прошлом
    now = datetime.now(timezone.utc)
    past_end = now - timedelta(minutes=5)
    
    trade = TradeHistory(
        id=1,
        market_id="m1",
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_price=0.25,
        market_end_time=past_end
    )
    
    # Сначала возвращаем настройки, потом список сделок
    db_mock.execute.return_value.scalar_one_or_none.side_effect = [mock_setting, None]
    db_mock.execute.return_value.scalars.return_value.all.return_value = [trade]

    trader_mock = MagicMock()
    api_mock = MagicMock()

    await stoploss_worker_cycle(db_mock, trader_mock, api_mock)

    assert trade.stop_loss_status == "EXPIRED"
    trader_mock.execute_trade.assert_not_called()
    assert db_mock.commit.call_count == 1


@pytest.mark.asyncio
async def test_worker_expired_when_market_missing_from_live():
    """Если рынок отсутствует в LiveMarket → EXPIRED."""
    db_mock = AsyncMock()
    mock_setting = MagicMock(key="STOP_LOSS_ENABLED", value="true")
    
    now = datetime.now(timezone.utc)
    future_end = now + timedelta(minutes=10)
    
    trade = TradeHistory(
        id=1,
        market_id="m1",
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_price=0.25,
        market_end_time=future_end
    )
    
    # Настройки, потом список сделок, потом LiveMarket (возвращаем None)
    db_mock.execute.return_value.scalar_one_or_none.side_effect = [mock_setting, None]
    db_mock.execute.return_value.scalars.return_value.all.return_value = [trade]

    trader_mock = MagicMock()
    api_mock = MagicMock()

    await stoploss_worker_cycle(db_mock, trader_mock, api_mock)

    assert trade.stop_loss_status == "EXPIRED"
    trader_mock.execute_trade.assert_not_called()
    assert db_mock.commit.call_count == 1


@pytest.mark.asyncio
async def test_worker_no_trigger_when_bid_above_stop():
    """При bid > stop_price sell не выполняется."""
    db_mock = AsyncMock()
    mock_setting = MagicMock(key="STOP_LOSS_ENABLED", value="true")
    
    now = datetime.now(timezone.utc)
    future_end = now + timedelta(minutes=10)
    
    trade = TradeHistory(
        id=1,
        market_id="m1",
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_pct=50.0,
        stop_loss_price=0.25,
        executed_price=0.50,
        market_end_time=future_end
    )
    
    market = LiveMarket(
        market_id="m1",
        yes_token_id="yes1",
        no_token_id="no1"
    )
    
    db_mock.execute.return_value.scalar_one_or_none.side_effect = [mock_setting, market]
    db_mock.execute.return_value.scalars.return_value.all.return_value = [trade]

    trader_mock = MagicMock()
    api_mock = AsyncMock()
    # bid = 0.30, что выше stop = 0.25
    api_mock.get_market_prices.return_value = {"best_bid": 0.30}

    await stoploss_worker_cycle(db_mock, trader_mock, api_mock)

    assert trade.stop_loss_status == "ACTIVE"
    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_worker_triggers_sell_when_bid_below_stop():
    """При bid <= stop_price выполняется sell-ордер и статус меняется на TRIGGERED."""
    db_mock = AsyncMock()
    mock_setting = MagicMock(key="STOP_LOSS_ENABLED", value="true")
    
    now = datetime.now(timezone.utc)
    future_end = now + timedelta(minutes=10)
    
    trade = TradeHistory(
        id=1,
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        status="SUCCESS",
        stop_loss_status="ACTIVE",
        stop_loss_pct=50.0,
        stop_loss_price=0.25,
        executed_price=0.50,
        market_end_time=future_end
    )
    
    market = LiveMarket(
        market_id="m1",
        yes_token_id="yes1",
        no_token_id="no1"
    )
    
    db_mock.execute.return_value.scalar_one_or_none.side_effect = [mock_setting, market]
    db_mock.execute.return_value.scalars.return_value.all.return_value = [trade]

    trader_mock = AsyncMock()
    trader_mock.execute_trade.return_value = {
        "status": "SUCCESS",
        "mode": "PAPER",
        "executed_price": 0.20
    }
    
    api_mock = AsyncMock()
    # bid = 0.20, что ниже stop = 0.25
    api_mock.get_market_prices.return_value = {"best_bid": 0.20}

    await stoploss_worker_cycle(db_mock, trader_mock, api_mock)

    assert trade.stop_loss_status == "TRIGGERED"
    trader_mock.execute_trade.assert_called_once_with(
        market_id="m1",
        token_id="yes1",
        side="SELL",
        price=0.20,
        size=20.0  # amount_usdc / executed_price = 10 / 0.50 = 20
    )
    # PnL = (0.20 - 0.50) * 20 = -6.0
    assert trade.pnl == -6.0
    assert db_mock.commit.call_count == 1
