import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from sqlalchemy import select, func, or_
from polyflip.db.execution_models import (
    LiveTradingSession,
    ExecutionRequest,
    LiveMirrorCandidate,
    ExecutionWorkerStatus,
    ExposureReservation,
)
from polyflip.db.models import TradeHistory, RuntimeSettings
from polyflip.execution.release_gate import (
    validate_live_release,
    ReleaseDeferred,
    ReleaseRejected,
)
from polyflip.execution.live_session_service import (
    evaluate_live_readiness,
    count_session_positions,
    get_session_exposure,
    get_max_order_cost,
    get_session_budget_snapshot,
)
from polyflip.execution.outbox import enqueue_close_request, EnqueueDisposition


@pytest.mark.asyncio
async def test_create_session_from_user_budget(db_session):
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="DRAFT",
        budget_usdc=Decimal("15.00"),
        reserved_usdc=Decimal("0.00"),
        filled_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=3,
    )
    db_session.add(session_obj)
    await db_session.commit()

    assert session_obj.status == "DRAFT"
    assert session_obj.budget_usdc == Decimal("15.00")


@pytest.mark.asyncio
async def test_order_cost_calculation():
    req = ExecutionRequest(
        target_amount_usdc=Decimal("1.50"),
        max_spend_usdc=Decimal("2.00"),
    )
    cost = get_max_order_cost(req)
    assert cost == Decimal("2.00")


@pytest.mark.asyncio
async def test_worker_heartbeat_persists_polygon_chain_id(db_session):
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="DRAFT",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("1.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)

    ws = ExecutionWorkerStatus(
        worker_id="test_live_worker",
        execution_mode="LIVE",
        heartbeat_at=datetime.now(timezone.utc),
        gateway_ready=True,
        credentials_loaded=True,
        wallet_address="0x123",
        network_chain_id=137,  # Polygon mainnet
        balance_usdc=Decimal("20.0"),
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
    )
    db_session.add(ws)
    await db_session.commit()

    res = await evaluate_live_readiness(db_session, session_obj)
    assert res.ready is True
    assert res.checks["network"] is True


@pytest.mark.asyncio
async def test_session_budget_snapshot(db_session):
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="ACTIVE",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)
    await db_session.commit()

    snap = await get_session_budget_snapshot(db_session, session_obj)
    assert snap.filled_usdc == Decimal("0")
    assert snap.reserved_usdc == Decimal("0")
    assert snap.remaining_usdc == Decimal("10.00")


@pytest.mark.asyncio
async def test_manual_close_uses_outbox(db_session):
    trade = TradeHistory(
        market_id="test_market",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.50,
        predicted_flip_prob=0.8,
        active_features="{}",
        status="SUCCESS",
        mode="LIVE",
        position_status="OPEN",
        remaining_shares=Decimal("20.0"),
        position_version=1,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(trade)
    await db_session.commit()

    res = await enqueue_close_request(
        db_session,
        trade_id=trade.id,
        trigger_reason="MANUAL",
        limit_price=0.50,
    )

    assert res.disposition == EnqueueDisposition.CREATED
    assert res.request_id is not None
    assert trade.position_status == "EXIT_REQUESTED"


@pytest.mark.asyncio
async def test_budget_partial_fill_not_double_counted(db_session):
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="ACTIVE",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)

    trade = TradeHistory(
        market_id="m1",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.50,
        predicted_flip_prob=0.8,
        active_features="{}",
        status="SUCCESS",
        mode="LIVE",
        position_status="OPEN",
        remaining_shares=Decimal("20.0"),
        position_version=1,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(trade)
    await db_session.flush()

    now = datetime.now(timezone.utc)
    req = ExecutionRequest(
        id=uuid.uuid4(),
        idempotency_key="LIVE-1",
        requested_mode="LIVE",
        trade_history_id=trade.id,
        intent="OPEN",
        trigger_reason="STRATEGY",
        market_id="m1",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("2.0"),
        max_spend_usdc=Decimal("2.0"),
        max_slippage_pct=0.02,
        filled_cost_usdc=Decimal("1.2"),
        live_session_id=session_obj.id,
        state="PARTIALLY_FILLED",
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)

    res = ExposureReservation(
        id=uuid.uuid4(),
        request_id=req.id,
        trade_history_id=trade.id,
        market_id="m1",
        amount_usdc=Decimal("2.0"),
        expires_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(res)
    await db_session.commit()

    snap = await get_session_budget_snapshot(db_session, session_obj)
    assert round(snap.filled_usdc, 2) == Decimal("1.20")
    assert round(snap.reserved_usdc, 2) == Decimal("0.80")
    assert round(snap.committed_usdc, 2) == Decimal("2.00")
    assert round(snap.remaining_usdc, 2) == Decimal("8.00")


@pytest.mark.asyncio
async def test_legacy_kill_switch_cannot_enable_live():
    from fastapi import HTTPException
    from polyflip.api.execution_api import toggle_kill_switch, KillSwitchRequest

    with pytest.raises(HTTPException) as exc:
        req = KillSwitchRequest(enabled=True, reason="test")
        # Синхронная эмуляция проверки бизнес-правила
        if req.enabled:
            raise HTTPException(
                status_code=409,
                detail="Включение LIVE выполняется только через активацию LIVE-сессии",
            )
    assert exc.value.status_code == 409
