import asyncio
import pytest
from uuid import uuid4
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt
from polyflip.db.models import LiveMarket

@pytest.mark.asyncio
async def test_worker_concurrency_skip_locked(db_session):
    if db_session.bind.dialect.name == "sqlite":
        pytest.skip("SQLite does not support row-level locks with FOR UPDATE SKIP LOCKED")
    import polyflip.db.connection
    import polyflip.execution.worker
    
    # Patch async_session globally for this test
    mock_sessionmaker = async_sessionmaker(db_session.bind, expire_on_commit=False)
    real_async_session = polyflip.db.connection.async_session
    polyflip.db.connection.async_session = mock_sessionmaker
    polyflip.execution.worker.async_session = mock_sessionmaker
    
    try:
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
            async with polyflip.db.connection.async_session() as session:
                stmt = select(ExecutionRequest).where(
                    ExecutionRequest.state == "READY"
                ).with_for_update(skip_locked=True)
                result = await session.execute(stmt)
                locked_req = result.scalar_one_or_none()
                
                await asyncio.sleep(2)
                
                if locked_req:
                    locked_req.state = "CLAIMED"
                    await session.commit()
        
        bg_task = asyncio.create_task(slow_worker_transaction())
        await asyncio.sleep(0.5)
        
        from polyflip.execution.worker import process_ready_requests
        await process_ready_requests()
        
        # Verify the request state hasn't been changed by the second call
        result = await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.id == req_id))
        current_req = result.scalar_one()
        assert current_req.state == "READY", "Second worker should not have modified the locked row"
        
        await bg_task
        
        result = await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.id == req_id))
        current_req = result.scalar_one()
        assert current_req.state == "CLAIMED", "First worker should have claimed the row"
    finally:
        polyflip.db.connection.async_session = real_async_session
        polyflip.execution.worker.async_session = real_async_session
