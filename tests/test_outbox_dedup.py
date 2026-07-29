import pytest
from sqlalchemy import select, func
from polyflip.execution.outbox import enqueue_open_request, EnqueueDisposition
from polyflip.execution.config import ExecutionMode
from polyflip.db.execution_models import ExecutionRequest
from polyflip.db.models import TradeHistory

from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_enqueue_open_request_dedup(db_session):
    """Два вызова enqueue_open_request с одинаковым (mode, market_id) должны создать ОДИН запрос."""
    trade = TradeHistory(
        market_id="BTC-USDC-TEST-DEDUP",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=1.0,
        executed_price=0.5,
        predicted_flip_prob=0.4,
        active_features="test",
        status="PENDING",
        mode="PAPER",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    res1 = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="BTC-USDC-TEST-DEDUP",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=1.0,
        limit_price=0.5,
        requested_mode=ExecutionMode.PAPER,
    )
    assert res1.disposition == EnqueueDisposition.CREATED
    assert res1.request_id is not None

    res2 = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="BTC-USDC-TEST-DEDUP",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=1.0,
        limit_price=0.5,
        requested_mode=ExecutionMode.PAPER,
    )
    assert res2.disposition == EnqueueDisposition.DUPLICATE
    assert res2.request_id == res1.request_id

    count = await db_session.scalar(
        select(func.count()).where(
            ExecutionRequest.market_id == "BTC-USDC-TEST-DEDUP",
            ExecutionRequest.intent == "OPEN",
        )
    )
    assert count == 1, f"Ожидался 1 запрос, найдено: {count}"
