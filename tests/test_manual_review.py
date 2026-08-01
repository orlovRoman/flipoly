import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
    ExecutionEvent,
    ExposureReservation,
)
from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility

from httpx import AsyncClient, ASGITransport


def make_trade():
    now = datetime.now(timezone.utc)
    return TradeHistory(
        market_id="test_market",
        asset="USDC",
        outcome_bought="YES",
        amount_usdc=Decimal("10.0"),
        predicted_flip_prob=0.6,
        active_features="test",
        model_version=1,
        edge=Decimal("0.1"),
        market_role="test",
        strategy_type="test",
        p_flip_effective=0.6,
        p_win_effective=0.6,
        stop_loss_pct=Decimal("0.1"),
        take_profit_enabled=False,
        take_profit_multiplier=Decimal("2.0"),
        mode="LIVE",
        status="PENDING",
        executed_price=0.0,
        position_status="OPENING",
        entry_filled_shares=Decimal("0"),
        entry_cost_usdc=Decimal("0"),
        remaining_shares=Decimal("0"),
        position_accounting_version=0,
        model_key="test",
        confirm_model_key="test",
        confirm_model_version=1,
        model_attribution_source="test",
        created_at=now,
        updated_at=now,
    )


def make_req(trade_id, mode="LIVE"):
    now = datetime.now(timezone.utc)
    return ExecutionRequest(
        id=uuid.uuid4(),
        trade_history_id=trade_id,
        intent="OPEN",
        trigger_reason="TEST",
        market_id="test_market",
        asset="USDC",
        outcome_to_buy="YES",
        state="MANUAL_REVIEW_REQUIRED",
        requested_mode=mode,
        target_amount_usdc=Decimal("10.0"),
        filled_shares=Decimal("0.0"),
        filled_cost_usdc=Decimal("0.0"),
        max_slippage_pct=Decimal("0.05"),
        ttl_seconds=30,
        expires_at=now,
        created_at=now,
        updated_at=now,
    )


def make_attempt(req_id, **kwargs):
    now = datetime.now(timezone.utc)
    base = {
        "id": uuid.uuid4(),
        "request_id": req_id,
        "gateway": "polymarket",
        "started_at": now,
        "attempt_no": 1,
        "status": "PENDING",
    }
    base.update(kwargs)
    return ExecutionAttempt(**base)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_allowed(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id)
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is True


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_provider_order_id(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, provider_order_id="0x")
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("provider_order_id" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_tx_hash(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, tx_hash="0x")
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("tx_hash" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_provider_trade_ids(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, provider_trade_ids=["t1"])
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("provider_trade_ids" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_transaction_hashes(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, transaction_hashes=["t1"])
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("transaction_hashes" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_execution_fill(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id)
    db_session.add(attempt)
    await db_session.flush()

    fill = ExecutionFill(
        id=uuid.uuid4(),
        attempt_id=attempt.id,
        provider_trade_id="x",
        shares=Decimal("1.0"),
        price=Decimal("0.5"),
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(fill)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("execution_fills" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_filled_shares(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    req.filled_shares = Decimal("1.0")
    db_session.add(req)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("заполненные shares" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_filled_cost_usdc(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    req.filled_cost_usdc = Decimal("1.0")
    db_session.add(req)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("стоимость исполнения" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_paper(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id, mode="PAPER")
    db_session.add(req)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("не относится к LIVE" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_mark_no_fill_releases_reservation(db_session):
    from polyflip.api.main import app

    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id)
    db_session.add(attempt)
    await db_session.flush()

    res = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req.id,
        trade_history_id=trade.id,
        market_id="test_market",
        amount_usdc=Decimal("10.0"),
        created_at=datetime.now(timezone.utc), expires_at=datetime.now(timezone.utc),
    )
    db_session.add(res)
    await db_session.commit()

    # Patch dependency to use test session
    from polyflip.db.connection import get_db_session
    from polyflip.api.auth import verify_api_key
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/api/execution/requests/{req.id}/resolve-review",
            json={
                "action": "MARK_FAILED_NO_FILL",
                "operator": "test",
                "note": "confirmed",
            },
        )
        assert response.status_code == 200

    await db_session.refresh(req)
    await db_session.refresh(trade)
    await db_session.refresh(res)

    assert req.state == "MANUAL_REVIEW_FAILED"
    assert trade.position_status == "ENTRY_FAILED"
    assert res.released_at is not None

    from sqlalchemy import select, func
    event_count = await db_session.scalar(select(func.count(ExecutionEvent.id)).where(ExecutionEvent.request_id == req.id))
    assert event_count >= 1

    app.dependency_overrides.clear()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_no_fill_resolution_is_idempotent(pg_session_factory):
    from polyflip.api.main import app
    import asyncio
    
    async with pg_session_factory() as db_session:
        trade = make_trade()
        db_session.add(trade)
        await db_session.flush()

        req = make_req(trade.id)
        db_session.add(req)
        await db_session.flush()

        attempt = make_attempt(req.id)
        db_session.add(attempt)
        await db_session.flush()

        res = ExposureReservation(
            id=uuid.uuid4(),
            request_id=req.id,
            trade_history_id=trade.id,
            market_id="test_market",
            amount_usdc=Decimal("10.0"),
            created_at=datetime.now(timezone.utc), expires_at=datetime.now(timezone.utc),
        )
        db_session.add(res)
        await db_session.commit()
        
        req_id = req.id
        trade_id = trade.id
        res_id = res.id

    # Need multiple clients hitting the DB directly
    async def make_request():
        # use a fresh session override per request for concurrency simulation
        from polyflip.db.connection import get_db_session
        from polyflip.api.auth import verify_api_key
        async with pg_session_factory() as temp_db:
            app.dependency_overrides[get_db_session] = lambda: temp_db
            app.dependency_overrides[verify_api_key] = lambda: True
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    f"/api/execution/requests/{req_id}/resolve-review",
                    json={
                        "action": "MARK_FAILED_NO_FILL",
                        "operator": "test",
                        "note": "confirmed",
                    },
                )
            app.dependency_overrides.clear()
            return resp

    responses = await asyncio.gather(
        make_request(), make_request(), make_request()
    )

    # Some might get 409 because state changed, some get 200
    statuses = [r.status_code for r in responses]
    assert 200 in statuses
    # The others could be 200 (if handled idempotently and we return 200) or 409
    
    async with pg_session_factory() as db_session:
        req = await db_session.get(ExecutionRequest, req_id)
        trade = await db_session.get(TradeHistory, trade_id)
        res = await db_session.get(ExposureReservation, res_id)
        
        assert req.state == "MANUAL_REVIEW_FAILED"
        assert trade.position_status == "ENTRY_FAILED"
        assert res.released_at is not None


        # Verify only 1 ExecutionEvent was created
        from sqlalchemy import select, func
        event_count = await db_session.scalar(
            select(func.count(ExecutionEvent.id))
            .where(ExecutionEvent.request_id == req_id)
            .where(ExecutionEvent.event_type == "MANUAL_REVIEW_MARK_FAILED_NO_FILL")
        )
        assert event_count == 1


@pytest.mark.asyncio
async def test_resolve_no_fill_batch_safe_and_unsafe(db_session):
    from polyflip.api.main import app
    from polyflip.db.connection import get_db_session
    from polyflip.api.auth import verify_api_key

    # Safe request
    trade1 = make_trade()
    db_session.add(trade1)
    await db_session.flush()

    req1 = make_req(trade1.id)
    db_session.add(req1)
    await db_session.flush()

    res1 = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req1.id,
        trade_history_id=trade1.id,
        market_id="test_market",
        amount_usdc=Decimal("10.0"),
        created_at=datetime.now(timezone.utc), expires_at=datetime.now(timezone.utc),
    )
    db_session.add(res1)
    await db_session.flush()

    # Unsafe request (has execution fills)
    trade2 = make_trade()
    db_session.add(trade2)
    await db_session.flush()

    req2 = make_req(trade2.id)
    req2.market_id = "test_market_2"
    db_session.add(req2)
    await db_session.flush()

    attempt = make_attempt(req2.id)
    db_session.add(attempt)
    await db_session.flush()

    fill = ExecutionFill(
        id=uuid.uuid4(),
        attempt_id=attempt.id,
        provider_trade_id="x",
        shares=Decimal("1.0"),
        price=Decimal("0.5"),
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(fill)
    await db_session.commit()

    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/execution/requests/resolve-no-fill-batch",
            json={
                "request_ids": [str(req1.id), str(req2.id)],
                "operator": "batch_op",
                "note": "batch_test"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert str(req1.id) in data["resolved"]
        assert any(s["request_id"] == str(req2.id) for s in data["skipped"])

    await db_session.refresh(req1)
    assert req1.state == "MANUAL_REVIEW_FAILED"

    await db_session.refresh(req2)
    assert req2.state == "MANUAL_REVIEW_REQUIRED"

    from sqlalchemy import select, func
    event_count = await db_session.scalar(
        select(func.count(ExecutionEvent.id))
        .where(ExecutionEvent.request_id == req1.id)
        .where(ExecutionEvent.event_type == "MANUAL_REVIEW_BATCH_NO_FILL")
    )
    assert event_count == 1
