import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy import select
from polyflip.db.models import TradeHistory, Asset
from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt
from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility
from polyflip.execution.outbox import process_execution_outbox

@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_allowed(session):
    # Setup trade history
    trade = TradeHistory(
        id=uuid.uuid4(),
        market_id="test_market",
        asset="USDC",
        mode="LIVE",
        intent="BUY_TO_OPEN",
        position_status="OPEN",
        amount_usdc=Decimal("10.0"),
        remaining_shares=Decimal("0.0"),
        created_at=datetime.now(timezone.utc)
    )
    session.add(trade)
    await session.flush()

    req = ExecutionRequest(
        id=uuid.uuid4(),
        trade_history_id=trade.id,
        intent="BUY_TO_OPEN",
        market_id="test_market",
        asset="USDC",
        state="MANUAL_REVIEW_REQUIRED",
        requested_mode="LIVE",
        filled_shares=Decimal("0.0"),
        filled_cost_usdc=Decimal("0.0"),
        created_at=datetime.now(timezone.utc)
    )
    session.add(req)
    await session.flush()

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req.id,
        started_at=datetime.now(timezone.utc),
        provider_order_id=None
    )
    session.add(attempt)
    await session.commit()

    eligibility = await evaluate_no_fill_eligibility(session, req)
    assert eligibility.allowed is True
    assert len(eligibility.blockers) == 0

@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_has_provider_id(session):
    # Setup trade history
    trade = TradeHistory(
        id=uuid.uuid4(),
        market_id="test_market",
        asset="USDC",
        mode="LIVE",
        intent="BUY_TO_OPEN",
        position_status="OPEN",
        amount_usdc=Decimal("10.0"),
        remaining_shares=Decimal("0.0"),
        created_at=datetime.now(timezone.utc)
    )
    session.add(trade)
    await session.flush()

    req = ExecutionRequest(
        id=uuid.uuid4(),
        trade_history_id=trade.id,
        intent="BUY_TO_OPEN",
        market_id="test_market",
        asset="USDC",
        state="MANUAL_REVIEW_REQUIRED",
        requested_mode="LIVE",
        filled_shares=Decimal("0.0"),
        filled_cost_usdc=Decimal("0.0"),
        created_at=datetime.now(timezone.utc)
    )
    session.add(req)
    await session.flush()

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req.id,
        started_at=datetime.now(timezone.utc),
        provider_order_id="0x_order_hash"
    )
    session.add(attempt)
    await session.commit()

    eligibility = await evaluate_no_fill_eligibility(session, req)
    assert eligibility.allowed is False
    assert any("has provider_order_id" in str(b) for b in eligibility.blockers)
