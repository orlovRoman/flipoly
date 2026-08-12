import pytest
import pickle
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone, timedelta
from sqlalchemy import select

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, SlippageLog
from polyflip.db.execution_models import ExecutionRequest
from polyflip.trading.takeprofit import compute_take_profit_price, evaluate_take_profit
from polyflip.trading.takeprofit_worker import takeprofit_worker_cycle


# ── Чистые unit-тесты (без БД) ─────────────────────────────────────────────

def test_tp_price_calculation():
    assert compute_take_profit_price(0.40, 2.0) == 0.80

def test_tp_price_x15():
    assert compute_take_profit_price(0.40, 1.5) == 0.60

def test_tp_price_above_market_max_is_disabled():
    # 0.60 * 2.0 = 1.20: the market resolves at $1.00, so no early TP.
    assert compute_take_profit_price(0.60, 2.0) is None


def test_tp_price_very_low_entry_is_clamped_to_minimum():
    assert compute_take_profit_price(0.001, 2.0) == 0.01

def test_tp_price_invalid_multiplier():
    with pytest.raises(ValueError):
        compute_take_profit_price(0.50, 0.9)

def test_evaluate_tp_triggers():
    d = evaluate_take_profit(entry_price=0.40, tp_multiplier=2.0, current_bid=0.82)
    assert d.should_sell is True
    assert d.tp_price == 0.80

def test_evaluate_tp_not_triggered():
    d = evaluate_take_profit(entry_price=0.40, tp_multiplier=2.0, current_bid=0.75)
    assert d.should_sell is False

def test_evaluate_tp_above_market_max_has_no_price():
    d = evaluate_take_profit(entry_price=0.60, tp_multiplier=2.0, current_bid=0.95)
    assert d.should_sell is False
    assert d.tp_price is None
    assert d.reason == "target_exceeds_max_price"


def test_evaluate_tp_exact_boundary():
    # current_bid == tp_price → должен сработать (>=)
    d = evaluate_take_profit(entry_price=0.40, tp_multiplier=2.0, current_bid=0.80)
    assert d.should_sell is True


# ── Integration-тесты с БД ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tp_worker_triggers_when_price_reached(db_session):
    """Воркер закрывает позицию и записывает TRIGGERED при достижении цели."""
    now = datetime.now(timezone.utc)

    db_session.add(RuntimeSettings(
        key="TAKE_PROFIT_ENABLED", value="true",
        updated_at=now, updated_by="test"
    ))
    db_session.add(LiveMarket(
        market_id="m_tp1", asset="BTC", question="Test?",
        yes_token_id="t_yes", no_token_id="t_no",
        current_yes_price=0.82, current_no_price=0.18,
        current_spread=0.01, volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(minutes=5), last_updated=now,
    ))
    trade = TradeHistory(
        market_id="m_tp1", asset="BTC",
        mode="PAPER", outcome_bought="YES",
        amount_usdc=10.0, remaining_shares=20.0, executed_price=0.40, status="SUCCESS",
        predicted_flip_prob=0.8, active_features="mid_price",
        take_profit_enabled=True, take_profit_multiplier=2.0,
        take_profit_price=0.80, take_profit_status="ACTIVE",
        market_end_time=now + timedelta(minutes=5),
        created_at=now,
    )
    db_session.add(trade)
    await db_session.commit()

    from tests.helpers import make_dummy_execution
    trader_mock = AsyncMock()
    trader_mock.execute_trade = AsyncMock(return_value=make_dummy_execution(
        mode="PAPER", executed_price=0.82, executed_usdc=8.2
    ))
    api_mock = AsyncMock()
    api_mock.get_market_prices = AsyncMock(return_value={
        "best_bid": 0.82, "current_yes_price": 0.82, "current_spread": 0.01
    })

    await takeprofit_worker_cycle(db_session, api_mock)

    result = await db_session.execute(
        select(TradeHistory).where(TradeHistory.market_id == "m_tp1")
    )
    t = result.scalar_one()
    assert t.take_profit_status == "QUEUED"
    assert t.status == "SUCCESS"
    
    from polyflip.db.execution_models import ExecutionRequest
    req = (await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.trade_history_id == t.id))).scalar_one_or_none()
    assert req is not None
    assert req.intent == "CLOSE"
    assert req.state == "READY"
    assert req.requested_mode == trade.mode

    

@pytest.mark.asyncio
async def test_live_tp_posts_gtd_request_immediately(db_session):
    """LIVE TP uses a resting GTD order and does not wait for the current bid."""
    now = datetime.now(timezone.utc)
    db_session.add_all([
        RuntimeSettings(key="TAKE_PROFIT_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TAKE_PROFIT_ORDER_MODE", value="GTD", updated_at=now, updated_by="test"),
        LiveMarket(
            market_id="m_tp_gtd", asset="BTC", question="Test?",
            yes_token_id="t_yes_gtd", no_token_id="t_no_gtd",
            current_yes_price=0.45, current_no_price=0.55, current_spread=0.01,
            volume_5min=100.0, price_velocity=0.0,
            end_time_est=now + timedelta(minutes=5), last_updated=now,
        ),
    ])
    trade = TradeHistory(
        market_id="m_tp_gtd", asset="BTC", mode="LIVE", outcome_bought="YES",
        amount_usdc=10.0, remaining_shares=20.0, executed_price=0.40, status="SUCCESS",
        predicted_flip_prob=0.8, active_features="mid_price",
        take_profit_enabled=True, take_profit_multiplier=2.0,
        take_profit_price=0.80, take_profit_status="ACTIVE",
        market_end_time=now + timedelta(minutes=5), created_at=now,
    )
    db_session.add(trade)
    await db_session.commit()

    api_mock = AsyncMock()
    await takeprofit_worker_cycle(db_session, api_mock)

    saved_trade = (await db_session.execute(
        select(TradeHistory).where(TradeHistory.market_id == "m_tp_gtd")
    )).scalar_one()
    assert saved_trade.take_profit_status == "QUEUED"
    req = (await db_session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == saved_trade.id)
    )).scalar_one()
    assert req.trigger_reason == "TAKE_PROFIT"
    assert float(req.limit_price) == pytest.approx(0.80)
    req_expiry = req.expires_at.replace(tzinfo=timezone.utc) if req.expires_at.tzinfo is None else req.expires_at
    assert req_expiry > now + timedelta(minutes=4)
    api_mock.get_market_prices.assert_not_called()

    # A second polling cycle must not enqueue a second close request.
    await takeprofit_worker_cycle(db_session, api_mock)
    requests = (await db_session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == saved_trade.id)
    )).scalars().all()
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_tp_worker_does_not_trigger_below_target(db_session):
    """Воркер не закрывает позицию, если цена ещё не достигла цели."""
    now = datetime.now(timezone.utc)

    db_session.add(RuntimeSettings(
        key="TAKE_PROFIT_ENABLED", value="true",
        updated_at=now, updated_by="test"
    ))
    db_session.add(LiveMarket(
        market_id="m_tp2", asset="BTC", question="Test?",
        yes_token_id="t_yes2", no_token_id="t_no2",
        current_yes_price=0.70, current_no_price=0.30,
        current_spread=0.01, volume_5min=50.0, price_velocity=0.0,
        end_time_est=now + timedelta(minutes=5), last_updated=now,
    ))
    trade = TradeHistory(
        market_id="m_tp2", asset="BTC",
        mode="PAPER", outcome_bought="YES",
        amount_usdc=10.0, remaining_shares=20.0, executed_price=0.40, status="SUCCESS",
        predicted_flip_prob=0.8, active_features="mid_price",
        take_profit_enabled=True, take_profit_multiplier=2.0,
        take_profit_price=0.80, take_profit_status="ACTIVE",
        market_end_time=now + timedelta(minutes=5),
        created_at=now,
    )
    db_session.add(trade)
    await db_session.commit()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()
    api_mock.get_market_prices = AsyncMock(return_value={
        "best_bid": 0.70, "current_yes_price": 0.70, "current_spread": 0.01
    })

    await takeprofit_worker_cycle(db_session, api_mock)

    result = await db_session.execute(
        select(TradeHistory).where(TradeHistory.market_id == "m_tp2")
    )
    t = result.scalar_one()
    assert t.take_profit_status == "ACTIVE"   # не тронуто
    assert t.pnl is None
    trader_mock.execute_trade.assert_not_called()


@pytest.mark.asyncio
async def test_tp_worker_skips_when_disabled(db_session):
    """Воркер не делает ничего, если TAKE_PROFIT_ENABLED=false."""
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(
        key="TAKE_PROFIT_ENABLED", value="false",
        updated_at=now, updated_by="test"
    ))
    await db_session.commit()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()

    await takeprofit_worker_cycle(db_session, api_mock)
    trader_mock.execute_trade.assert_not_called()
    api_mock.get_market_prices.assert_not_called()


@pytest.mark.asyncio
async def test_tp_market_expired_sets_status(db_session):
    """Воркер ставит EXPIRED если рынок уже закрыт."""
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(
        key="TAKE_PROFIT_ENABLED", value="true",
        updated_at=now, updated_by="test"
    ))
    trade = TradeHistory(
        market_id="m_expired", asset="BTC",
        mode="PAPER", outcome_bought="YES",
        amount_usdc=10.0, remaining_shares=20.0, executed_price=0.40, status="SUCCESS",
        predicted_flip_prob=0.8, active_features="mid_price",
        take_profit_enabled=True, take_profit_multiplier=2.0,
        take_profit_price=0.80, take_profit_status="ACTIVE",
        market_end_time=now - timedelta(minutes=5),  # уже истёк
        created_at=now - timedelta(minutes=10),
    )
    db_session.add(trade)
    await db_session.commit()

    trader_mock = AsyncMock()
    api_mock = AsyncMock()
    await takeprofit_worker_cycle(db_session, api_mock)

    result = await db_session.execute(
        select(TradeHistory).where(TradeHistory.market_id == "m_expired")
    )
    t = result.scalar_one()
    assert t.take_profit_status == "EXPIRED"
    trader_mock.execute_trade.assert_not_called()
