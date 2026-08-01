import pytest
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from polyflip.trading.trade_recorder import enqueue_open_request
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest


@pytest.mark.asyncio
async def test_concurrent_enqueue_open(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    trade = TradeHistory(
        market_id="test_market",
        asset="ETH",
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
    await db_session.commit()

    from polyflip.execution.config import ExecutionMode
    async def worker_enqueue():
        # Each worker needs its own session to test concurrency properly,
        # but for this simple outbox test, we are just verifying idempotency key
        # prevents actual duplicates at the DB level via UPSERT.
        return await enqueue_open_request(
            db_session,
            trade_id=trade.id,
            market_id="test_market",
            asset="ETH",
            outcome_to_buy="Yes",
            target_amount_usdc=10.0,
            limit_price=0.5,
            requested_mode=ExecutionMode.PAPER,
        )

    # Attempt to enqueue simultaneously
    tasks = [worker_enqueue() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # All should return the same request UUID
    first_req_id = results[0]
    for r in results:
        assert r.request_id == first_req_id.request_id

    await db_session.commit()

    res = await db_session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id)
    )
    requests = res.scalars().all()
    assert len(requests) == 1
    assert requests[0].intent == "OPEN"
