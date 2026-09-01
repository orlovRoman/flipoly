from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from polyflip.db.execution_models import ExecutionAttempt, ExecutionRequest
from polyflip.db.models import LiveMarket, TradeHistory
from polyflip.execution.config import ExecutionMode
from polyflip.execution.outbox import enqueue_open_request


def _trade(market_id: str) -> TradeHistory:
    return TradeHistory(
        market_id=market_id,
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=1.0,
        executed_price=0.25,
        predicted_flip_prob=0.60,
        active_features="{}",
        status="PENDING",
        mode="PAPER",
        position_status="OPENING",
        position_accounting_version=0,
        position_version=0,
        created_at=datetime.now(timezone.utc),
    )


def _market(market_id: str) -> LiveMarket:
    now = datetime.now(timezone.utc)
    return LiveMarket(
        market_id=market_id,
        asset="BTC",
        question="BTC Up or Down?",
        yes_token_id="YES_TOKEN",
        no_token_id="NO_TOKEN",
        end_time_est=now + timedelta(minutes=15),
        current_yes_price=0.25,
        current_no_price=0.75,
        current_spread=0.01,
        last_updated=now,
    )


@pytest.mark.asyncio
async def test_process_ready_request_finishes_paper_trade(
    engine, db_session, monkeypatch
):
    """The real worker path must finish READY -> FILLED on the same DB session."""
    import polyflip.execution.worker as worker

    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("PAPER_EXECUTION_PROFILE", "INSTANT")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(worker, "async_session", session_factory)

    market_id = "paper-worker-e2e"
    trade = _trade(market_id)
    db_session.add_all([trade, _market(market_id)])
    await db_session.flush()
    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=market_id,
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=1.0,
        limit_price=0.25,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.commit()
    trade_id = trade.id

    await worker.process_ready_requests()

    db_session.expire_all()
    request = await db_session.get(ExecutionRequest, result.request_id)
    persisted_trade = await db_session.get(TradeHistory, trade_id)

    assert request.state == "FILLED"
    assert request.filled_shares == Decimal("4")
    assert persisted_trade.status == "SUCCESS"
    assert persisted_trade.position_status == "OPEN"
    assert persisted_trade.remaining_shares == Decimal("4")


@pytest.mark.asyncio
async def test_submit_error_requeues_paper_instead_of_leaving_submitting(
    engine, db_session, monkeypatch
):
    import polyflip.execution.worker as worker

    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("PAPER_EXECUTION_PROFILE", "INSTANT")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(worker, "async_session", session_factory)

    async def broken_submit(_self, _order):
        raise RuntimeError("synthetic persistence failure")

    monkeypatch.setattr(
        "polyflip.execution.gateways.fake.FakeExecutionGateway.submit",
        broken_submit,
    )

    market_id = "paper-worker-retry"
    trade = _trade(market_id)
    db_session.add_all([trade, _market(market_id)])
    await db_session.flush()
    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=market_id,
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=1.0,
        limit_price=0.25,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.commit()

    await worker.process_ready_requests()

    db_session.expire_all()
    request = await db_session.get(ExecutionRequest, result.request_id)
    attempts = (
        (
            await db_session.execute(
                select(ExecutionAttempt).where(
                    ExecutionAttempt.request_id == result.request_id
                )
            )
        )
        .scalars()
        .all()
    )

    assert request.state == "READY"
    assert request.claimed_by is None
    assert "PAPER retry" in request.error_reason
    assert len(attempts) == 1
    assert attempts[0].status == "FAILED"


@pytest.mark.asyncio
async def test_reconcile_requeues_stale_paper_without_provider_id(
    engine, db_session, monkeypatch
):
    import polyflip.execution.worker as worker

    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("PAPER_EXECUTION_PROFILE", "INSTANT")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(worker, "async_session", session_factory)

    market_id = "paper-worker-stale"
    trade = _trade(market_id)
    db_session.add_all([trade, _market(market_id)])
    await db_session.flush()
    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=market_id,
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=1.0,
        limit_price=0.25,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.flush()

    request = await db_session.get(ExecutionRequest, result.request_id)
    request.state = "SUBMITTING"
    request.updated_at = datetime.now(timezone.utc) - timedelta(seconds=90)
    attempt = ExecutionAttempt(
        request_id=request.id,
        gateway="FAKE",
        attempt_no=1,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=90),
    )
    db_session.add(attempt)
    await db_session.commit()
    request_id = request.id
    attempt_id = attempt.id

    await worker.reconcile_active_requests()

    db_session.expire_all()
    recovered = await db_session.get(ExecutionRequest, request_id)
    recovered_attempt = await db_session.get(ExecutionAttempt, attempt_id)

    assert recovered.state == "READY"
    assert recovered_attempt.status == "FAILED"
    assert recovered_attempt.provider_order_id is None
