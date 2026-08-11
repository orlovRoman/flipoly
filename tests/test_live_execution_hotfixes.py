import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExposureReservation,
)
from polyflip.db.models import TradeHistory
from polyflip.execution.release_gate import (
    _build_live_trade,
    validate_live_release,
    _build_signal_snapshot,
    _compute_hash,
)
from polyflip.execution.worker import _finish_submit_exception
from polyflip.api.execution_api import resolve_manual_review, ResolveReviewRequest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_minimum_order_rejection_is_terminal(db_session):
    """Детерминированный отказ (GatewayOrderRejected) переводит заявку в REJECTED и освобождает резерв."""
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()
    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        entry_filled_shares=Decimal("0"),
        entry_cost_usdc=Decimal("0"),
        remaining_shares=Decimal("0"),
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MIRROR",
        state="CLAIMED",
        trade_history_id=trade.id,
        market_id="m1",
        asset="BTC",
        outcome_to_buy="UP",
        requested_shares=Decimal("2.0"),
        target_amount_usdc=Decimal("1.00"),
        max_slippage_pct=0.02,
        limit_price=Decimal("0.5"),
        max_spend_usdc=Decimal("1.00"),
        ttl_seconds=30,
        expires_at=now + timedelta(seconds=30),
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    res = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req_id,
        trade_history_id=trade.id,
        market_id="m1",
        amount_usdc=Decimal("1.00"),
        expires_at=now + timedelta(seconds=30),
        created_at=now,
    )
    db_session.add(res)

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req_id,
        gateway="POLYMARKET",
        attempt_no=1,
        status="RUNNING",
        started_at=now,
    )
    db_session.add(attempt)
    await db_session.commit()

    await _finish_submit_exception(
        db_session,
        request_id=req_id,
        attempt_id=attempt.id,
        attempt_no=1,
        requested_mode="LIVE",
        error="Order rejected: invalid amount for a marketable BUY order ($0.97), min size: 1",
        is_deterministic_rejection=True,
    )

    await db_session.refresh(req)
    await db_session.refresh(trade)
    await db_session.refresh(res)

    assert req.state == "REJECTED"
    assert trade.position_status == "ENTRY_FAILED"
    assert trade.remaining_shares == Decimal("0")
    assert res.released_at is not None


@pytest.mark.asyncio
async def test_unknown_network_result_requires_review(db_session):
    """Сетевая неопределенность переводит заявку в MANUAL_REVIEW_REQUIRED, удерживая резерв."""
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()
    trade = TradeHistory(
        market_id="m2",
        asset="ETH",
        outcome_bought="DOWN",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MIRROR",
        state="CLAIMED",
        trade_history_id=trade.id,
        market_id="m2",
        asset="ETH",
        outcome_to_buy="DOWN",
        requested_shares=Decimal("2.0"),
        target_amount_usdc=Decimal("1.00"),
        max_slippage_pct=0.02,
        limit_price=Decimal("0.5"),
        max_spend_usdc=Decimal("1.00"),
        ttl_seconds=30,
        expires_at=now + timedelta(seconds=30),
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    res = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req_id,
        trade_history_id=trade.id,
        market_id="m2",
        amount_usdc=Decimal("1.00"),
        expires_at=now + timedelta(seconds=30),
        created_at=now,
    )
    db_session.add(res)

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req_id,
        gateway="POLYMARKET",
        attempt_no=1,
        status="RUNNING",
        started_at=now,
    )
    db_session.add(attempt)
    await db_session.commit()

    await _finish_submit_exception(
        db_session,
        request_id=req_id,
        attempt_id=attempt.id,
        attempt_no=1,
        requested_mode="LIVE",
        error="Submission unknown: ConnectionTerminated",
        is_deterministic_rejection=False,
    )

    await db_session.refresh(req)
    await db_session.refresh(res)

    assert req.state == "MANUAL_REVIEW_REQUIRED"
    assert res.released_at is None


@pytest.mark.asyncio
async def test_no_open_position_before_fill():
    """При запуске кандидата сделка создаётся со статусом PENDING и OPENING."""
    paper_trade = MagicMock()
    paper_trade.id = 100
    paper_trade.market_id = "m_test"
    paper_trade.asset = "BTC"
    paper_trade.outcome_bought = "UP"
    paper_trade.amount_usdc = Decimal("1.00")
    paper_trade.predicted_flip_prob = 0.6
    paper_trade.active_features = "{}"
    paper_trade.model_version = 1
    paper_trade.edge = 0.05
    paper_trade.market_role = "FAVORITE"
    paper_trade.strategy_type = "FLIP"
    paper_trade.p_flip_effective = 0.6
    paper_trade.p_win_effective = 0.6
    paper_trade.stop_loss_pct = 0.1
    paper_trade.stop_loss_price = 0.4
    paper_trade.take_profit_enabled = False
    paper_trade.take_profit_multiplier = 2.0
    paper_trade.take_profit_price = None
    paper_trade.model_key = "k"
    paper_trade.confirm_model_key = None
    paper_trade.confirm_model_version = None
    paper_trade.model_attribution_source = "s"
    paper_trade.config_snapshot = {}
    paper_trade.market_end_time = None

    now = datetime.now(timezone.utc)
    candidate = MagicMock()

    trade = _build_live_trade(candidate, paper_trade, now, "LIVE", Decimal("1.00"))

    assert trade.status == "PENDING"
    assert trade.position_status == "OPENING"
    assert trade.entry_filled_shares == Decimal("0")
    assert trade.remaining_shares == Decimal("0")
    assert trade.executed_price == 0.0


@pytest.mark.asyncio
async def test_mark_failed_no_fill_releases_reservation(db_session):
    """MARK_FAILED_NO_FILL переводит заявку в MANUAL_REVIEW_FAILED и освобождает резерв."""
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()
    trade = TradeHistory(
        market_id="m3",
        asset="DOGE",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MIRROR",
        state="MANUAL_REVIEW_REQUIRED",
        trade_history_id=trade.id,
        market_id="m3",
        asset="DOGE",
        outcome_to_buy="UP",
        requested_shares=Decimal("2.0"),
        target_amount_usdc=Decimal("1.00"),
        max_slippage_pct=0.02,
        limit_price=Decimal("0.5"),
        max_spend_usdc=Decimal("1.00"),
        ttl_seconds=30,
        expires_at=now + timedelta(seconds=30),
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    res = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req_id,
        trade_history_id=trade.id,
        market_id="m3",
        amount_usdc=Decimal("1.00"),
        expires_at=now + timedelta(seconds=30),
        created_at=now,
    )
    db_session.add(res)

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req_id,
        gateway="POLYMARKET",
        attempt_no=1,
        status="FAILED",
        provider_order_id=None,
        started_at=now,
    )
    db_session.add(attempt)
    await db_session.commit()

    body = ResolveReviewRequest(
        action="MARK_FAILED_NO_FILL",
        operator="test_admin",
        note="Confirmed no fill",
    )

    resp = await resolve_manual_review(str(req_id), body, db_session)

    await db_session.refresh(req)
    await db_session.refresh(trade)
    await db_session.refresh(res)

    assert resp["new_state"] == "MANUAL_REVIEW_FAILED"
    assert req.state == "MANUAL_REVIEW_FAILED"
    assert trade.position_status == "ENTRY_FAILED"
    assert res.released_at is not None

    from polyflip.db.execution_models import ExecutionEvent

    events = (
        (
            await db_session.execute(
                select(ExecutionEvent)
                .where(ExecutionEvent.request_id == req.id)
                .order_by(ExecutionEvent.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    assert len(events) >= 1
    event_types = [e.event_type for e in events]
    assert "MANUAL_REVIEW_MARK_FAILED_NO_FILL" in event_types

    target_event = next(
        e for e in events if e.event_type == "MANUAL_REVIEW_MARK_FAILED_NO_FILL"
    )
    assert target_event.created_at is not None


@pytest.mark.asyncio
async def test_mark_failed_no_fill_forbidden_if_provider_order_id(db_session):
    """MARK_FAILED_NO_FILL сбрасывает HTTP 422 если есть provider_order_id."""
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()
    trade = TradeHistory(
        market_id="m4",
        asset="SOL",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MIRROR",
        state="MANUAL_REVIEW_REQUIRED",
        trade_history_id=trade.id,
        market_id="m4",
        asset="SOL",
        outcome_to_buy="UP",
        requested_shares=Decimal("2.0"),
        target_amount_usdc=Decimal("1.00"),
        max_slippage_pct=0.02,
        limit_price=Decimal("0.5"),
        max_spend_usdc=Decimal("1.00"),
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req_id,
        gateway="POLYMARKET",
        attempt_no=1,
        status="FAILED",
        provider_order_id="0xPROVIDER_ORDER_123",
        started_at=now,
    )
    db_session.add(attempt)
    await db_session.commit()

    body = ResolveReviewRequest(
        action="MARK_FAILED_NO_FILL",
        operator="test_admin",
        note="Attempting to force fail",
    )

    with pytest.raises(HTTPException) as exc_info:
        await resolve_manual_review(str(req_id), body, db_session)

    assert exc_info.value.status_code == 409
    assert "provider_order_id" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_minimum_live_order_amount_validation():
    """Валидатор release_gate отклоняет LIVE ордер меньше 1.10 USDC."""
    trade_id = 999
    req_id = uuid.uuid4()

    paper_req = MagicMock()
    paper_req.id = req_id
    paper_req.trade_history_id = trade_id
    paper_req.requested_mode = "PAPER"
    paper_req.intent = "OPEN"
    paper_req.state = "FILLED"
    paper_req.target_amount_usdc = Decimal("1.00")
    paper_req.created_at = datetime.now(timezone.utc)
    paper_req.updated_at = datetime.now(timezone.utc)
    paper_req.asset = "BTC"
    paper_req.market_id = "m_test"
    paper_req.outcome_to_buy = "UP"
    paper_req.limit_price = Decimal("0.5")
    paper_req.max_spend_usdc = Decimal("1.00")

    paper_trade = MagicMock()
    paper_trade.id = trade_id
    paper_trade.mode = "PAPER"
    paper_trade.asset = "BTC"
    paper_trade.market_id = "m_test"
    paper_trade.outcome_bought = "UP"
    paper_trade.market_end_time = datetime.now(timezone.utc) + timedelta(minutes=10)

    snap = _build_signal_snapshot(paper_req, paper_trade)
    sig_hash = _compute_hash(snap)

    candidate = MagicMock()
    candidate.target_mode = "LIVE"
    candidate.signal_hash = sig_hash

    db_session = AsyncMock()
    db_session.scalar.side_effect = ["60", "true", 0, 0, 0, 0, 0, 0, 0, 0]

    active_session_mock = MagicMock()
    active_session_mock.max_single_order_usdc = Decimal("10.0")
    active_session_mock.budget_usdc = Decimal("100.0")
    active_session_mock.max_open_positions = 5
    active_session_mock.max_total_exposure_usdc = Decimal("50.0")
    active_session_mock.order_amount_usdc = None
    active_session_mock.heartbeat_at = datetime.now(timezone.utc)
    active_session_mock.gateway_ready = True
    active_session_mock.collateral_allowance_ready = True
    active_session_mock.conditional_allowance_ready = True
    active_session_mock.balance_usdc = Decimal("100.0")

    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = active_session_mock
    execute_result.one.return_value = (Decimal("0"), Decimal("0"), Decimal("0"))
    db_session.execute.return_value = execute_result

    with patch(
        "polyflip.execution.release_gate.check_risk_limits", new_callable=AsyncMock
    ) as mock_risk:
        mock_risk.return_value = None
        plan = await validate_live_release(
            db_session, candidate, paper_req, paper_trade, "LIVE", fresh_prices={"best_ask": 0.5}
        )
        assert plan.order_amount_usdc == Decimal("1.10")


@pytest.mark.asyncio
async def test_reconcile_manual_review_with_provider_id(db_session):
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()

    trade = TradeHistory(
        market_id="m1",
        asset="ETH",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MANUAL",
        state="MANUAL_REVIEW_REQUIRED",
        trade_history_id=trade.id,
        market_id="m1",
        asset="ETH",
        outcome_to_buy="UP",
        requested_shares=Decimal("1.0"),
        target_amount_usdc=Decimal("1.0"),
        max_spend_usdc=Decimal("1.0"),
        max_slippage_pct=0.02,
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req_id,
        gateway="POLYMARKET",
        attempt_no=1,
        status="FAILED",
        provider_order_id="0xPROVIDER_ORDER_123",
        started_at=now,
    )
    db_session.add(attempt)
    await db_session.commit()

    from polyflip.api.execution_api import reconcile_request

    res = await reconcile_request(req_id, db_session)

    assert res["state"] == "RECONCILING"
    await db_session.refresh(req)
    assert req.state == "RECONCILING"


@pytest.mark.asyncio
async def test_reconcile_without_provider_id_rejected(db_session):
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()

    trade = TradeHistory(
        market_id="m1",
        asset="ETH",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MANUAL",
        state="MANUAL_REVIEW_REQUIRED",
        trade_history_id=trade.id,
        market_id="m1",
        asset="ETH",
        outcome_to_buy="UP",
        requested_shares=Decimal("1.0"),
        target_amount_usdc=Decimal("1.0"),
        max_spend_usdc=Decimal("1.0"),
        max_slippage_pct=0.02,
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)
    await db_session.commit()

    from polyflip.api.execution_api import reconcile_request

    with pytest.raises(HTTPException) as exc:
        await reconcile_request(req_id, db_session)

    assert exc.value.status_code == 422
    assert "provider_order_id" in exc.value.detail


@pytest.mark.asyncio
async def test_reconcile_filled_request_rejected(db_session):
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()

    trade = TradeHistory(
        market_id="m1",
        asset="ETH",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MANUAL",
        state="FILLED",
        trade_history_id=trade.id,
        market_id="m1",
        asset="ETH",
        outcome_to_buy="UP",
        requested_shares=Decimal("1.0"),
        target_amount_usdc=Decimal("1.0"),
        max_spend_usdc=Decimal("1.0"),
        max_slippage_pct=0.02,
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)
    await db_session.commit()

    from polyflip.api.execution_api import reconcile_request

    with pytest.raises(HTTPException) as exc:
        await reconcile_request(req_id, db_session)

    assert exc.value.status_code == 409
    assert "нельзя переводить в RECONCILING" in exc.value.detail


@pytest.mark.asyncio
async def test_serializer_combines_no_fill_and_reconcile_actions(db_session):
    now = datetime.now(timezone.utc)
    req_id = uuid.uuid4()

    trade = TradeHistory(
        market_id="m1",
        asset="ETH",
        outcome_bought="UP",
        amount_usdc=Decimal("1.00"),
        executed_price=0.0,
        predicted_flip_prob=0.5,
        active_features="{}",
        model_version=1,
        status="PENDING",
        mode="LIVE",
        position_status="OPENING",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    req = ExecutionRequest(
        id=req_id,
        idempotency_key=f"LIVE-OPEN-{req_id}",
        requested_mode="LIVE",
        intent="OPEN",
        trigger_reason="MANUAL",
        state="MANUAL_REVIEW_REQUIRED",
        trade_history_id=trade.id,
        market_id="m1",
        asset="ETH",
        outcome_to_buy="UP",
        requested_shares=Decimal("1.0"),
        target_amount_usdc=Decimal("1.0"),
        max_spend_usdc=Decimal("1.0"),
        max_slippage_pct=0.02,
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    attempt = ExecutionAttempt(
        id=uuid.uuid4(),
        request_id=req_id,
        gateway="POLYMARKET",
        attempt_no=1,
        status="FAILED",
        provider_order_id="0x123",
        started_at=now,
    )
    db_session.add(attempt)
    await db_session.commit()

    from polyflip.execution.serializers import serialize_execution_requests

    with patch(
        "polyflip.execution.serializers.evaluate_no_fill_eligibility_batch",
        new_callable=AsyncMock,
    ) as mock_check:
        mock_eligibility = MagicMock()
        mock_eligibility.allowed = True
        mock_eligibility.blockers = []
        mock_check.return_value = {req_id: mock_eligibility}

        results = await serialize_execution_requests(db_session, [req])
        print("\nREQ ID:", req.id)
        print("MOCK RETURN:", mock_check.return_value)
        print("CALL ARGS:", mock_check.call_args)

    assert len(results) == 1
    actions = results[0]["available_actions"]
    assert "MARK_FAILED_NO_FILL" in actions
    assert "RECONCILE_WITH_POLYMARKET" in actions
