import asyncio
import pytest
from uuid import uuid4
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy import select

from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt
from polyflip.db.models import LiveMarket
from polyflip.execution.worker import process_ready_requests

@pytest.mark.asyncio
async def test_worker_concurrency_skip_locked(db_session):
    # Prepare a market and a READY ExecutionRequest
    market = LiveMarket(
        market_id="test_market_concurrent",
        asset="BTC",
        question="BTC > 100k?",
        yes_token_id="yes_tok",
        no_token_id="no_tok",
        end_time_est=datetime.now(timezone.utc),
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        last_updated=datetime.now(timezone.utc)
    )
    db_session.add(market)
    
    req_id = uuid4()
    req = ExecutionRequest(
        id=req_id,
        intent="OPEN",
        market_id="test_market_concurrent",
        asset="BTC",
        outcome_to_buy="YES",
        requested_shares=Decimal("10.0"),
        target_amount_usdc=Decimal("5.0"),
        max_slippage_pct=2.0,
        ttl_seconds=60,
        state="READY",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req)
    await db_session.commit()

    # Create a long-running transaction simulating a slow worker holding the row lock
    async def slow_worker_transaction():
        from polyflip.db.connection import async_session
        async with async_session() as session:
            # We acquire the lock
            stmt = select(ExecutionRequest).where(
                ExecutionRequest.state == "READY"
            ).with_for_update(skip_locked=True)
            result = await session.execute(stmt)
            locked_req = result.scalar_one_or_none()
            
            # Now we hold the lock, let's wait to allow another worker to try
            await asyncio.sleep(2)
            
            if locked_req:
                locked_req.state = "CLAIMED"
                await session.commit()
    
    # Run the slow transaction in the background
    bg_task = asyncio.create_task(slow_worker_transaction())
    
    # Give the background task a moment to acquire the lock
    await asyncio.sleep(0.5)
    
    # Now try to run process_ready_requests() which should skip the locked row and return None (since it's the only row)
    from polyflip.execution.worker import process_ready_requests
    await process_ready_requests()
    
    # Verify the request state hasn't been changed by the second call
    result = await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.id == req_id))
    current_req = result.scalar_one()
    assert current_req.state == "READY", "Second worker should not have modified the locked row"
    
    # Wait for the slow transaction to finish
    await bg_task
    
    # Now the state should be CLAIMED
    result = await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.id == req_id))
    current_req = result.scalar_one()
    assert current_req.state == "CLAIMED", "First worker should have claimed the row"
