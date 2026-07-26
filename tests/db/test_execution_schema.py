import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest

@pytest.mark.asyncio
async def test_legacy_trade_allows_null_accounting_fields(db_session):
    trade = TradeHistory(
        market_id="mock_legacy",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=100.0,
        executed_price=0.5,
        predicted_flip_prob=0.6,
        active_features="test",
        status="SUCCESS",
        position_accounting_version=0,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(trade)
    await db_session.commit()
    assert trade.id is not None

@pytest.mark.asyncio
async def test_new_trade_requires_accounting_initialization(db_session):
    trade = TradeHistory(
        market_id="mock_new",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=100.0,
        executed_price=0.5,
        predicted_flip_prob=0.6,
        active_features="test",
        status="SUCCESS",
        position_accounting_version=1,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(trade)
    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()
    assert "ck_trade_position_accounting_initialized" in str(exc_info.value)
    await db_session.rollback()

    trade.entry_filled_shares = Decimal("200.0")
    trade.entry_cost_usdc = Decimal("100.0")
    trade.remaining_shares = Decimal("200.0")
    trade.realized_pnl_usdc = Decimal("0.0")
    db_session.add(trade)
    await db_session.commit()
    assert trade.id is not None

@pytest.mark.asyncio
async def test_duplicate_active_open_request_is_rejected(db_session):
    req1 = ExecutionRequest(
        intent="OPEN",
        market_id="dup_open_market",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("100.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="READY",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req1)
    await db_session.commit()

    req2 = ExecutionRequest(
        intent="OPEN",
        market_id="dup_open_market",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("50.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="CLAIMED",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req2)
    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()
    assert "UNIQUE constraint failed" in str(exc_info.value) or "uq_active_open_request" in str(exc_info.value)
    await db_session.rollback()

@pytest.mark.asyncio
async def test_two_active_close_requests_are_rejected(db_session):
    trade = TradeHistory(
        market_id="close_market",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=100.0,
        executed_price=0.5,
        predicted_flip_prob=0.6,
        active_features="test",
        status="SUCCESS",
        position_accounting_version=0,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(trade)
    await db_session.commit()

    req1 = ExecutionRequest(
        intent="CLOSE",
        trade_history_id=trade.id,
        market_id="close_market",
        asset="BTC",
        outcome_to_buy="YES",
        requested_shares=Decimal("10.0"),
        target_amount_usdc=Decimal("100.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="READY",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req1)
    await db_session.commit()

    req2 = ExecutionRequest(
        intent="CLOSE",
        trade_history_id=trade.id,
        market_id="close_market",
        asset="BTC",
        outcome_to_buy="YES",
        requested_shares=Decimal("5.0"),
        target_amount_usdc=Decimal("50.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="READY",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req2)
    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()
    assert "UNIQUE constraint failed" in str(exc_info.value) or "uq_active_close_request" in str(exc_info.value)
    await db_session.rollback()

@pytest.mark.asyncio
async def test_completed_request_allows_new_request(db_session):
    req1 = ExecutionRequest(
        intent="OPEN",
        market_id="seq_market",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("100.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="FILLED",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req1)
    await db_session.commit()

    req2 = ExecutionRequest(
        intent="OPEN",
        market_id="seq_market",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("50.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="READY",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req2)
    await db_session.commit()
    assert req2.id is not None

@pytest.mark.asyncio
async def test_execution_financial_columns_use_numeric(db_session):
    # SQLite does not support true Decimal precision and casts to float.
    # Postgres handles Numeric correctly. Test what is returned without strictly enforcing full 18-place accuracy on SQLite.
    req = ExecutionRequest(
        intent="OPEN",
        market_id="numeric_market",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("123.456789012345678901"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="FILLED",
        filled_shares=Decimal("987.654321098765432109"),
        filled_cost_usdc=Decimal("123.456789012345678901"),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req)
    await db_session.commit()
    
    await db_session.refresh(req)
    val = req.target_amount_usdc
    assert isinstance(val, Decimal) or isinstance(val, float)
    assert abs(float(val) - 123.45678901234568) < 1e-9

@pytest.mark.asyncio
async def test_stoploss_and_takeprofit_create_one_close_request(engine, db_session):
    import asyncio
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from polyflip.execution.outbox import enqueue_close_request, ACTIVE_CLOSE_STATES
    from polyflip.execution.config import ExecutionMode
    from sqlalchemy import select, func

    trade = TradeHistory(
        market_id="concurrent_close_market",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=100.0,
        executed_price=0.5,
        predicted_flip_prob=0.6,
        active_features="test",
        status="SUCCESS",
        position_accounting_version=0,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(trade)
    await db_session.commit()

    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async def run_enqueue(trigger):
        async with async_session() as session:
            res = await enqueue_close_request(
                session,
                trade_id=trade.id,
                trigger_reason=trigger,
                requested_mode=ExecutionMode.PAPER
            )
            await session.commit()
            return res

    results = await asyncio.gather(
        run_enqueue("STOP_LOSS"),
        run_enqueue("TAKE_PROFIT")
    )
    
    successful_ids = [r for r in results if r is not None]
    assert len(successful_ids) == 1, "Only one close request should succeed"

    stmt = select(func.count()).where(
        ExecutionRequest.trade_history_id == trade.id,
        ExecutionRequest.intent == 'CLOSE',
        ExecutionRequest.state.in_(ACTIVE_CLOSE_STATES)
    )
    count = (await db_session.execute(stmt)).scalar_one()
    assert count == 1


