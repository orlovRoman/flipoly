import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
from polyflip.execution.contracts import GatewayReadiness, BalanceResult


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
async def test_worker_readiness_persists_polygon_chain_id(db_session):
    """Тест: refresh_gateway_readiness_once считывает готовность из gateway и сохраняет network_chain_id=137 в БД."""
    from polyflip.execution.worker import refresh_gateway_readiness_once

    gateway = AsyncMock()
    gateway.get_readiness.return_value = GatewayReadiness(
        ready=True,
        gateway="POLYMARKET",
        wallet_address="0x1234567890abcdef",
        balance=BalanceResult(
            balance_usdc=Decimal("20.0"),
            collateral_allowances={},
            conditional_allowances_checked=2,
            conditional_allowance_ready=True,
            checked_at=datetime.now(timezone.utc),
        ),
        credentials_loaded=True,
        client_initialized=True,
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
        network_chain_id=137,
        checked_at=datetime.now(timezone.utc),
    )

    # Need a worker status first for readiness to update
    db_session.add(
        ExecutionWorkerStatus(
            worker_id="test_worker_1",
            execution_mode="LIVE",
            heartbeat_at=datetime.now(timezone.utc),
        )
    )
    await db_session.commit()

    await refresh_gateway_readiness_once(
        db_session,
        worker_id="test_worker_1",
        execution_mode="LIVE",
        gateway=gateway,
    )

    status = (
        await db_session.execute(
            select(ExecutionWorkerStatus).where(
                ExecutionWorkerStatus.execution_mode == "LIVE"
            )
        )
    ).scalar_one_or_none()

    assert status is not None
    assert status.network_chain_id == 137
    assert status.conditional_allowance_ready is True
    assert status.gateway_ready is True
    gateway.get_readiness.assert_awaited_once()

    # 2-й вызов readiness — проверяем обновление ВСЕХ полей
    gateway.get_readiness.return_value = GatewayReadiness(
        ready=False,
        gateway="POLYMARKET",
        wallet_address="0xNEW",
        balance=BalanceResult(
            balance_usdc=Decimal("7.5"),
            collateral_allowances={},
            conditional_allowances_checked=2,
            conditional_allowance_ready=False,
            checked_at=datetime.now(timezone.utc),
        ),
        credentials_loaded=False,
        client_initialized=True,
        collateral_allowance_ready=False,
        conditional_allowance_ready=False,
        network_chain_id=80002,
        error_message="wrong network",
        checked_at=datetime.now(timezone.utc),
    )

    await refresh_gateway_readiness_once(
        db_session,
        worker_id="test_worker_1",
        execution_mode="LIVE",
        gateway=gateway,
    )

    await db_session.refresh(status)

    assert status.gateway_ready is False
    assert status.credentials_loaded is True
    assert status.wallet_address == "0x1234567890abcdef"
    assert status.balance_usdc == 20.0
    assert status.network_chain_id == 137
    assert status.conditional_allowance_ready is True
    assert status.last_error_message == "wrong network"
    assert gateway.get_readiness.await_count == 2


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
async def test_legacy_kill_switch_cannot_enable_live(db_session):
    """Тест: эндпоинт toggle_kill_switch с enabled=True сбрасывает HTTP 409."""
    from fastapi import HTTPException
    from polyflip.api.execution_api import toggle_kill_switch, KillSwitchRequest

    payload = KillSwitchRequest(enabled=True, reason="test_enable")
    with pytest.raises(HTTPException) as exc_info:
        await toggle_kill_switch(payload, db_session)

    assert exc_info.value.status_code == 409
    assert "Включение LIVE выполняется только через активацию LIVE-сессии" in str(
        exc_info.value.detail
    )


def test_live_mode_requires_relayer_credentials(monkeypatch):
    from polyflip.execution.config import ExecutionSettings

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("POLYGON_PRIVATE_KEY", "0xprivate")
    monkeypatch.setenv("POLYGON_ADDRESS", "0xwallet")
    monkeypatch.delenv("POLYMARKET_RELAYER_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_RELAYER_API_KEY_ADDRESS", raising=False)

    with pytest.raises(ValueError, match="POLYMARKET_RELAYER_API_KEY"):
        ExecutionSettings()


@pytest.mark.asyncio
async def test_activation_sets_live_mirror_started_at(db_session):
    from polyflip.api.execution_api import activate_live_session
    from polyflip.db.execution_models import LiveTradingSession
    from polyflip.db.models import RuntimeSettings
    from sqlalchemy import select

    session_id = uuid.uuid4()
    session_obj = LiveTradingSession(
        id=session_id,
        status="READY",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)
    await db_session.commit()

    with patch("polyflip.api.execution_api.evaluate_live_readiness") as mock_eval:
        mock_eval.return_value = MagicMock(ready=True, errors=[])
        res = await activate_live_session(str(session_id), db_session)

    assert res["status"] == "ACTIVE"

    row = (
        await db_session.execute(
            select(RuntimeSettings).where(
                RuntimeSettings.key == "LIVE_MIRROR_STARTED_AT"
            )
        )
    ).scalar_one_or_none()

    assert row is not None
    assert row.value != ""
    assert row.updated_by == "session_activate"


@pytest.mark.asyncio
async def test_calculate_live_order_amount():
    from polyflip.execution.release_gate import (
        calculate_live_order_amount,
        ReleaseDeferred,
    )
    from polyflip.db.execution_models import LiveTradingSession, ExecutionRequest
    from decimal import Decimal

    session = LiveTradingSession(
        max_single_order_usdc=Decimal("10.00"), order_amount_usdc=Decimal("5.00")
    )
    req = ExecutionRequest(
        target_amount_usdc=Decimal("1.50"),
        max_spend_usdc=Decimal("1.50"),
    )

    # Should use session.order_amount_usdc
    amt = calculate_live_order_amount(req, session)
    assert amt == Decimal("5.00")

    # Fallback to PAPER size if order_amount_usdc is None
    session.order_amount_usdc = None
    amt2 = calculate_live_order_amount(req, session)
    assert amt2 == Decimal("1.50")

    # Fallback with LIVE_MIN_GROSS_BUY_USDC
    req.target_amount_usdc = Decimal("0.50")
    req.max_spend_usdc = Decimal("0.50")
    amt3 = calculate_live_order_amount(req, session)
    assert amt3 == Decimal("1.10")  # Minimum from config

    # Exceeds max_single_order_usdc
    session.order_amount_usdc = Decimal("15.00")
    with pytest.raises(ReleaseDeferred):
        calculate_live_order_amount(req, session)


@pytest.mark.asyncio
async def test_evaluate_live_readiness_ssl_error(db_session):
    from polyflip.execution.live_session_service import evaluate_live_readiness

    # Setup active session and worker
    session = LiveTradingSession(
        id=uuid.uuid4(),
        status="READY",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session)

    worker = ExecutionWorkerStatus(
        worker_id="test-1",
        execution_mode="LIVE",
        heartbeat_at=datetime.now(timezone.utc),
        gateway_ready=True,
        credentials_loaded=True,
        balance_usdc=10.0,
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
        last_error_code="TLS_TRANSPORT_ERROR",
        last_error_message="SSL SYSCALL error: EOF detected",
    )
    db_session.add(worker)
    await db_session.commit()

    res = await evaluate_live_readiness(db_session, session)
    assert not res.ready
    assert any("временно недоступен" in e for e in res.errors)
    assert any("SSL SYSCALL error" in e for e in res.errors)
    assert any(
        "Баланс и approvals показаны по последней успешной проверке" in w
        for w in res.warnings
    )


@pytest.mark.asyncio
async def test_patch_live_session_limits_endpoint(db_session):
    from polyflip.api.execution_api import (
        update_live_session_limits,
        UpdateLiveSessionLimitsRequest,
    )

    session = LiveTradingSession(
        id=uuid.uuid4(),
        status="DRAFT",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session)
    await db_session.commit()

    payload = UpdateLiveSessionLimitsRequest(
        order_amount_usdc=Decimal("3.00"),
        max_single_order_usdc=Decimal("4.00"),
        budget_usdc=Decimal("20.00"),
    )

    res = await update_live_session_limits(str(session.id), payload, db_session)

    assert res["order_amount_usdc"] == 3.0
    assert res["max_single_order_usdc"] == 4.0
    assert res["budget_usdc"] == 20.0
    assert res["status"] == "DRAFT"


@pytest.mark.asyncio
async def test_transport_failure_always_blocks_live_activation(db_session):
    from polyflip.execution.live_session_service import evaluate_live_readiness

    session = LiveTradingSession(
        id=uuid.uuid4(),
        status="READY",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session)

    worker = ExecutionWorkerStatus(
        worker_id="test-1",
        execution_mode="LIVE",
        heartbeat_at=datetime.now(timezone.utc),
        gateway_ready=True,
        credentials_loaded=True,
        balance_usdc=10.0,
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
        last_error_code="NETWORK_TRANSPORT_ERROR",
        last_error_message="Connection reset by peer",
    )
    db_session.add(worker)
    await db_session.commit()

    res = await evaluate_live_readiness(db_session, session)
    assert not res.ready
    assert any("временно недоступен" in e for e in res.errors)


@pytest.mark.asyncio
async def test_patch_live_session_limits_cases_negative(db_session):
    from polyflip.api.execution_api import (
        update_live_session_limits,
        UpdateLiveSessionLimitsRequest,
    )
    from fastapi import HTTPException

    session = LiveTradingSession(
        id=uuid.uuid4(),
        status="DRAFT",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session)
    await db_session.commit()

    payload = UpdateLiveSessionLimitsRequest(
        budget_usdc=Decimal("-5.00"),
    )

    with pytest.raises(HTTPException) as excinfo:
        await update_live_session_limits(str(session.id), payload, db_session)

    assert excinfo.value.status_code == 422
    assert "Бюджет должен быть больше нуля" in str(excinfo.value.detail)
