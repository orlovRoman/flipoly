import pytest
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from polyflip.execution.outbox import enqueue_open_request, enqueue_close_request
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest


@pytest.mark.asyncio
async def test_enqueue_open_idempotent(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    trade = TradeHistory(
        market_id="test_market",
        asset="BTC",
        outcome_bought="Yes",
        executed_price=0.0,
        strategy_type="LIGHTGBM_TREND",
        predicted_flip_prob=0.8,
        market_role="FAVORITE",
        position_status="OPEN",
        status="PENDING",
        amount_usdc=10.0,
        position_accounting_version=1,
        position_version=1,
        active_features="LIGHTGBM_TREND",
        mode="PAPER",
        entry_filled_shares=0.0,
        entry_cost_usdc=0.0,
        remaining_shares=0.0,
        realized_pnl_usdc=0.0,
        created_at=now,
    )
    db_session.add(trade)
    await db_session.flush()

    from polyflip.execution.config import ExecutionMode

    req_id_1 = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="test_market",
        asset="BTC",
        outcome_to_buy="Yes",
        target_amount_usdc=10.0,
        limit_price=0.5,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.commit()

    req_id_2 = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="test_market",
        asset="BTC",
        outcome_to_buy="Yes",
        target_amount_usdc=10.0,
        limit_price=0.5,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.commit()

    assert req_id_1.request_id == req_id_2.request_id
    assert req_id_1.disposition == "CREATED"
    assert req_id_2.disposition != "CREATED"
    res = await db_session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id)
    )
    requests = res.scalars().all()
    assert len(requests) == 1
    assert requests[0].intent == "OPEN"


@pytest.mark.asyncio
async def test_enqueue_close_idempotent(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    trade = TradeHistory(
        market_id="test_market",
        asset="BTC",
        outcome_bought="Yes",
        executed_price=0.0,
        strategy_type="LIGHTGBM_TREND",
        predicted_flip_prob=0.8,
        market_role="FAVORITE",
        position_status="OPEN",
        status="SUCCESS",
        amount_usdc=10.0,
        remaining_shares=20.0,
        position_accounting_version=1,
        position_version=1,
        active_features="LIGHTGBM_TREND",
        mode="PAPER",
        entry_filled_shares=20.0,
        entry_cost_usdc=10.0,
        realized_pnl_usdc=0.0,
        created_at=now,
    )
    db_session.add(trade)
    await db_session.flush()

    from polyflip.execution.config import ExecutionMode

    req_id_1 = await enqueue_close_request(
        db_session,
        trade_id=trade.id,
        trigger_reason="TAKE_PROFIT",
        limit_price=0.6,
    )
    await db_session.commit()

    req_id_2 = await enqueue_close_request(
        db_session,
        trade_id=trade.id,
        trigger_reason="TAKE_PROFIT",
        limit_price=0.6,
    )
    await db_session.commit()

    assert req_id_1.request_id == req_id_2.request_id
    assert req_id_1.disposition == "CREATED"
    assert req_id_2.disposition != "CREATED"
    res = await db_session.execute(
        select(ExecutionRequest).where(
            ExecutionRequest.trade_history_id == trade.id,
            ExecutionRequest.intent == "CLOSE",
        )
    )
    requests = res.scalars().all()
    assert len(requests) == 1
