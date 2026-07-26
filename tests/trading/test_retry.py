import pytest
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from polyflip.trading.trade_recorder import enqueue_open_request, enqueue_close_request
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest, ExecutionIntent


@pytest.mark.asyncio
async def test_enqueue_open_idempotent(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    trade = TradeHistory(
        asset="BTC",
        direction="UP",
        strategy_type="LIGHTGBM_TREND",
        predicted_flip_prob=0.8,
        market_role="FAVORITE",
        position_status="OPEN",
        status="PENDING",
        amount_usdc=10.0,
        position_accounting_version=1,
        position_version=1,
    )
    db_session.add(trade)
    await db_session.flush()

    req_id_1 = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="test_market",
        token_id="test_token",
        bet_size_usdc=10.0,
        limit_price=0.5,
    )
    await db_session.commit()

    req_id_2 = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="test_market",
        token_id="test_token",
        bet_size_usdc=10.0,
        limit_price=0.5,
    )
    await db_session.commit()

    assert req_id_1 == req_id_2
    res = await db_session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id)
    )
    requests = res.scalars().all()
    assert len(requests) == 1
    assert requests[0].intent == ExecutionIntent.OPEN


@pytest.mark.asyncio
async def test_enqueue_close_idempotent(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    trade = TradeHistory(
        asset="BTC",
        direction="UP",
        strategy_type="LIGHTGBM_TREND",
        predicted_flip_prob=0.8,
        market_role="FAVORITE",
        position_status="OPEN",
        status="SUCCESS",
        amount_usdc=10.0,
        remaining_shares=20.0,
        position_accounting_version=1,
        position_version=1,
    )
    db_session.add(trade)
    await db_session.flush()

    req_id_1 = await enqueue_close_request(
        db_session,
        trade_id=trade.id,
        trigger_reason="TAKE_PROFIT",
        market_id="test_market",
        token_id="test_token",
        shares_to_sell=20.0,
        limit_price=0.6,
        position_version=1,
    )
    await db_session.commit()

    req_id_2 = await enqueue_close_request(
        db_session,
        trade_id=trade.id,
        trigger_reason="TAKE_PROFIT",
        market_id="test_market",
        token_id="test_token",
        shares_to_sell=20.0,
        limit_price=0.6,
        position_version=1,
    )
    await db_session.commit()

    assert req_id_1 == req_id_2
    res = await db_session.execute(
        select(ExecutionRequest).where(
            ExecutionRequest.trade_history_id == trade.id,
            ExecutionRequest.intent == ExecutionIntent.CLOSE
        )
    )
    requests = res.scalars().all()
    assert len(requests) == 1
