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
from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket
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

    # Need a live market and worker status first for readiness to update
    db_session.add(
        LiveMarket(
            market_id="test_m1",
            asset="BTC",
            question="BTC up?",
            yes_token_id="probe_token_123",
            no_token_id="probe_token_456",
            end_time_est=datetime.now(timezone.utc) + timedelta(hours=1),
            current_yes_price=0.5,
            current_no_price=0.5,
            current_spread=0.01,
            last_updated=datetime.now(timezone.utc),
        )
    )
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
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("budget_usdc", Decimal("-5"), "Бюджет"),
        ("max_total_exposure_usdc", Decimal("-5"), "экспозиция"),
        ("max_single_order_usdc", Decimal("-5"), "ставка"),
    ],
)
async def test_patch_live_session_limits_cases_negative(db_session, field, value, message):
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

    kwargs = {field: value}
    payload = UpdateLiveSessionLimitsRequest(**kwargs)

    with pytest.raises(HTTPException) as excinfo:
        await update_live_session_limits(str(session.id), payload, db_session)

    assert excinfo.value.status_code == 422
    assert message in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_readiness_uses_one_global_approval_probe(db_session):
    from polyflip.execution.worker import refresh_gateway_readiness_once

    now = datetime.now(timezone.utc)
    for i in range(70):
        db_session.add(
            LiveMarket(
                market_id=f"market_{i}",
                asset="BTC",
                question=f"Question {i}",
                yes_token_id=f"yes_token_{i}",
                no_token_id=f"no_token_{i}",
                end_time_est=now + timedelta(hours=1),
                current_yes_price=0.5,
                current_no_price=0.5,
                current_spread=0.01,
                last_updated=now,
            )
        )

    db_session.add(
        ExecutionWorkerStatus(
            worker_id="LIVE:test:1",
            execution_mode="LIVE",
            heartbeat_at=now,
        )
    )
    await db_session.commit()

    mock_gateway = AsyncMock()
    mock_gateway.get_readiness.return_value = GatewayReadiness(
        ready=True,
        gateway="POLYMARKET",
        wallet_address="0x334b3732225B105d1764E257CAcc5Bf939fc6a9D",
        balance=BalanceResult(
            balance_usdc=Decimal("67.88709"),
            collateral_allowances={},
            conditional_allowances_checked=1,
            conditional_allowance_ready=True,
            checked_at=now,
        ),
        credentials_loaded=True,
        client_initialized=True,
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
        network_chain_id=137,
        checked_at=now,
    )

    await refresh_gateway_readiness_once(
        db_session,
        worker_id="LIVE:test:1",
        execution_mode="LIVE",
        gateway=mock_gateway,
    )

    args = mock_gateway.get_readiness.await_args.kwargs
    assert len(args["conditional_token_ids"]) == 1


@pytest.mark.asyncio
async def test_readiness_persists_success_with_single_probe(db_session):
    from polyflip.execution.worker import refresh_gateway_readiness_once

    now = datetime.now(timezone.utc)
    db_session.add(
        LiveMarket(
            market_id="probe_market_1",
            asset="ETH",
            question="ETH up?",
            yes_token_id="probe_yes_token",
            no_token_id="probe_no_token",
            end_time_est=now + timedelta(hours=2),
            current_yes_price=0.5,
            current_no_price=0.5,
            current_spread=0.01,
            last_updated=now,
        )
    )
    db_session.add(
        ExecutionWorkerStatus(
            worker_id="LIVE:test:1",
            execution_mode="LIVE",
            heartbeat_at=now,
        )
    )
    await db_session.commit()

    mock_gateway = AsyncMock()
    mock_gateway.get_readiness.return_value = GatewayReadiness(
        ready=True,
        gateway="POLYMARKET",
        wallet_address="0x334b3732225B105d1764E257CAcc5Bf939fc6a9D",
        balance=BalanceResult(
            balance_usdc=Decimal("67.88709"),
            collateral_allowances={},
            conditional_allowances_checked=1,
            conditional_allowance_ready=True,
            checked_at=now,
        ),
        credentials_loaded=True,
        client_initialized=True,
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
        network_chain_id=137,
        checked_at=now,
    )

    await refresh_gateway_readiness_once(
        db_session,
        worker_id="LIVE:test:1",
        execution_mode="LIVE",
        gateway=mock_gateway,
    )

    status = await db_session.get(
        ExecutionWorkerStatus,
        "LIVE:test:1",
    )

    assert status.gateway_ready is True
    assert status.credentials_loaded is True
    assert round(status.balance_usdc, 5) == Decimal("67.88709")
    assert status.collateral_allowance_ready is True
    assert status.conditional_allowance_ready is True
    assert status.network_chain_id == 137
    assert status.last_error_code is None


@pytest.mark.asyncio
@patch("polyflip.collector.client.PolymarketClient")
@patch("polyflip.execution.worker.build_execution_gateway")
async def test_fak_no_liquidity_releases_reservation(mock_build_gateway, mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    from unittest.mock import AsyncMock
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    from polyflip.execution.worker import process_ready_requests
    from polyflip.execution.contracts import GatewayOrderRejected
    from polyflip.db.execution_models import ExecutionRequest, ExposureReservation
    from polyflip.db.models import TradeHistory, LiveMarket

    now = datetime.now(timezone.utc)
    db_session.add(
        LiveMarket(
            market_id="test_market",
            asset="BTC",
            question="BTC up?",
            yes_token_id="token-yes",
            no_token_id="token-no",
            end_time_est=now + timedelta(hours=2),
            current_yes_price=0.5,
            current_no_price=0.5,
            current_spread=0.01,
            last_updated=now,
        )
    )

    trade = TradeHistory(
        market_id="test_market",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=5.0,
        executed_price=0.5,
        predicted_flip_prob=0.8,
        active_features="test",
        status="PENDING",
        position_status="OPENING",
        mode="LIVE",
        created_at=now,
        updated_at=now,
    )
    db_session.add(trade)
    await db_session.flush()

    req = ExecutionRequest(
        id=uuid.uuid4(),
        trade_history_id=trade.id,
        intent="OPEN",
        target_amount_usdc=Decimal("5"),
        max_spend_usdc=Decimal("5"),
        limit_price=Decimal("0.5"),
        max_slippage_pct=2.0,
        requested_mode="LIVE",
        outcome_to_buy="YES",
        market_id="test_market",
        asset="BTC",
        idempotency_key="test-key",
        state="READY",
        created_at=now,
        updated_at=now,
    )
    db_session.add(req)
    await db_session.flush()

    reservation = ExposureReservation(
        request_id=req.id,
        trade_history_id=trade.id,
        market_id="test_market",
        amount_usdc=Decimal("5"),
        expires_at=now + timedelta(seconds=60),
    )
    db_session.add(reservation)
    await db_session.commit()

    mock_gateway = AsyncMock()
    mock_gateway.name = "POLYMARKET"
    mock_gateway.submit.side_effect = GatewayOrderRejected("NO_LIQUIDITY_FAK")
    mock_build_gateway.return_value = mock_gateway

    import contextlib

    @contextlib.asynccontextmanager
    async def mock_async_session():
        yield db_session

    with patch("polyflip.execution.worker.async_session", new=mock_async_session), \
         patch("polyflip.execution.worker.ExecutionSettings") as MockSettings, \
         patch("polyflip.execution.worker.check_risk_limits", return_value=None):
         
        MockSettings.return_value.execution_mode.value = "LIVE"
        await process_ready_requests()

    await db_session.refresh(req)
    await db_session.refresh(reservation)
    await db_session.refresh(trade)

    assert req.state == "REJECTED"
    assert req.filled_shares == Decimal("0")
    assert reservation.released_at is not None
    assert trade.position_status == "ENTRY_FAILED"


@pytest.mark.asyncio
@patch("polyflip.collector.client.PolymarketClient")
@patch("polyflip.execution.worker.build_execution_gateway")
async def test_reconcile_matched_order_fetches_fills_before_get_order(mock_build_gateway, mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    from unittest.mock import AsyncMock
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    from polyflip.execution.worker import reconcile_active_requests
    from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt
    from polyflip.db.models import LiveMarket
    from polyflip.execution.contracts import TradeExecution
    import uuid
    from datetime import datetime, timezone, timedelta
    from decimal import Decimal
    
    now = datetime.now(timezone.utc)
    market = LiveMarket(
        market_id="m1", asset="BTC", question="Q", yes_token_id="tok-yes", no_token_id="tok-no",
        end_time_est=now+timedelta(hours=2), current_yes_price=0.5, current_no_price=0.5,
        current_spread=0.01, last_updated=now
    )
    db_session.add(market)

    trade = TradeHistory(market_id="m1", asset="BTC", outcome_bought="YES", amount_usdc=5.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="test", status="PENDING", position_status="OPENING", mode="LIVE", created_at=now, updated_at=now)
    db_session.add(trade)
    await db_session.flush()
    
    req = ExecutionRequest(
        id=uuid.uuid4(), trade_history_id=trade.id, intent="OPEN", target_amount_usdc=Decimal("5"), max_spend_usdc=Decimal("5"),
        limit_price=Decimal("0.5"), max_slippage_pct=2.0, requested_mode="LIVE", outcome_to_buy="YES",
        market_id="m1", asset="BTC", state="RECONCILING", idempotency_key="k1", created_at=now-timedelta(minutes=5),
        updated_at=now-timedelta(minutes=5)
    )
    db_session.add(req)
    await db_session.flush()
    
    attempt = ExecutionAttempt(
        request_id=req.id, gateway="POLYMARKET", attempt_no=1, provider_order_id="ext-order-1", provider_status="MATCHED", status="UNKNOWN",
        started_at=now-timedelta(minutes=5)
    )
    db_session.add(attempt)
    await db_session.commit()
    
    mock_gateway = AsyncMock()
    mock_gateway.name = "POLYMARKET"
    mock_build_gateway.return_value = mock_gateway
    mock_gateway.fetch_order_fills.return_value = (
        TradeExecution(provider_trade_id="t1", gateway="POLYMARKET", gross_quote_usdc=Decimal("5"), price=Decimal("0.5"),
                       shares=Decimal("10"), fee_usdc=Decimal("0"), matched_at=now, transaction_hash="0xabc"),
    )
    
    import contextlib
    @contextlib.asynccontextmanager
    async def mock_async_session():
        yield db_session

    with patch("polyflip.execution.worker.async_session", new=mock_async_session), \
         patch("polyflip.execution.worker.ExecutionSettings") as MockSettings, \
         patch("polyflip.execution.worker.rebuild_trade_accounting", return_value=None):
        MockSettings.return_value.execution_mode.value = "LIVE"
        await reconcile_active_requests()
        
    mock_gateway.get_order.assert_not_called()
    await db_session.refresh(req)
    assert req.state == "FILLED"
    assert req.filled_shares == Decimal("10")
    
@pytest.mark.asyncio
@patch("polyflip.collector.client.PolymarketClient")
@patch("polyflip.execution.worker.build_execution_gateway")
async def test_reconcile_recovers_when_get_order_schema_is_invalid(mock_build_gateway, mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    from unittest.mock import AsyncMock
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    from polyflip.execution.worker import reconcile_active_requests
    from polyflip.db.execution_models import ExecutionRequest, ExecutionAttempt
    from polyflip.db.models import LiveMarket, TradeHistory
    from polyflip.execution.contracts import TradeExecution
    import uuid
    from datetime import datetime, timezone, timedelta
    from decimal import Decimal
    
    now = datetime.now(timezone.utc)
    market = LiveMarket(
        market_id="m2", asset="BTC", question="Q", yes_token_id="tok-yes", no_token_id="tok-no",
        end_time_est=now+timedelta(hours=2), current_yes_price=0.5, current_no_price=0.5,
        current_spread=0.01, last_updated=now
    )
    db_session.add(market)

    trade = TradeHistory(market_id="m2", asset="BTC", outcome_bought="YES", amount_usdc=5.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="test", status="PENDING", position_status="OPENING", mode="LIVE", created_at=now, updated_at=now)
    db_session.add(trade)
    await db_session.flush()
    
    req = ExecutionRequest(
        id=uuid.uuid4(), trade_history_id=trade.id, intent="OPEN", target_amount_usdc=Decimal("5"), max_spend_usdc=Decimal("5"),
        limit_price=Decimal("0.5"), max_slippage_pct=2.0, requested_mode="LIVE", outcome_to_buy="YES",
        market_id="m2", asset="BTC", state="RECONCILING", idempotency_key="k2", created_at=now-timedelta(minutes=5),
        updated_at=now-timedelta(minutes=5)
    )
    db_session.add(req)
    await db_session.flush()
    
    attempt = ExecutionAttempt(
        request_id=req.id, gateway="POLYMARKET", attempt_no=1, provider_order_id="ext-order-2", provider_status="MATCHED", status="UNKNOWN",
        started_at=now-timedelta(minutes=5)
    )
    db_session.add(attempt)
    await db_session.commit()
    
    mock_gateway = AsyncMock()
    mock_gateway.name = "POLYMARKET"
    mock_build_gateway.return_value = mock_gateway
    mock_gateway.get_order.side_effect = Exception("UnexpectedResponseError: OpenOrder response did not match expected shape")
    mock_gateway.fetch_order_fills.return_value = (
        TradeExecution(provider_trade_id="t2", gateway="POLYMARKET", gross_quote_usdc=Decimal("5"), price=Decimal("0.5"),
                       shares=Decimal("10"), fee_usdc=Decimal("0"), matched_at=now, transaction_hash="0xdef"),
    )
    
    import contextlib
    @contextlib.asynccontextmanager
    async def mock_async_session():
        yield db_session

    with patch("polyflip.execution.worker.async_session", new=mock_async_session), \
         patch("polyflip.execution.worker.ExecutionSettings") as MockSettings, \
         patch("polyflip.execution.worker.rebuild_trade_accounting", return_value=None):
        MockSettings.return_value.execution_mode.value = "LIVE"
        await reconcile_active_requests()
        
    await db_session.refresh(req)
    assert req.state == "FILLED"
    assert req.filled_shares == Decimal("10")
    
@pytest.mark.asyncio
@patch("polyflip.collector.client.PolymarketClient")
@patch("polyflip.execution.worker.build_execution_gateway")
async def test_fak_no_liquidity_becomes_rejected_no_fill(mock_build_gateway, mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    from unittest.mock import AsyncMock
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    from polyflip.execution.worker import process_ready_requests
    from polyflip.execution.contracts import GatewayOrderRejected
    from polyflip.db.execution_models import ExecutionRequest, ExposureReservation
    from polyflip.db.models import TradeHistory, LiveMarket
    import uuid
    from datetime import datetime, timezone, timedelta
    from decimal import Decimal
    
    now = datetime.now(timezone.utc)
    market = LiveMarket(
        market_id="m3", asset="BTC", question="Q", yes_token_id="tok-yes", no_token_id="tok-no",
        end_time_est=now+timedelta(hours=2), current_yes_price=0.5, current_no_price=0.5,
        current_spread=0.01, last_updated=now
    )
    db_session.add(market)
    
    trade = TradeHistory(market_id="m3", asset="BTC", outcome_bought="YES", amount_usdc=5.0, executed_price=0.5, predicted_flip_prob=0.8, active_features="test", status="PENDING", position_status="OPENING", mode="LIVE", created_at=now, updated_at=now)
    db_session.add(trade)
    await db_session.flush()
    
    req = ExecutionRequest(id=uuid.uuid4(), trade_history_id=trade.id, intent="OPEN", target_amount_usdc=Decimal("5"), max_spend_usdc=Decimal("5"), limit_price=Decimal("0.5"), max_slippage_pct=2.0, requested_mode="LIVE", outcome_to_buy="YES", market_id="m3", asset="BTC", idempotency_key="k3", state="READY", created_at=now, updated_at=now)
    db_session.add(req)
    await db_session.flush()
    
    reservation = ExposureReservation(request_id=req.id, trade_history_id=trade.id, market_id="m3", amount_usdc=Decimal("5"), expires_at=now+timedelta(seconds=60))
    db_session.add(reservation)
    await db_session.commit()
    
    mock_gateway = AsyncMock()
    mock_gateway.name = "POLYMARKET"
    mock_build_gateway.return_value = mock_gateway
    mock_gateway.submit.side_effect = GatewayOrderRejected("NO_LIQUIDITY_FAK: FAK-заявка не нашла встречной ликвидности и не была исполнена")
    
    import contextlib
    @contextlib.asynccontextmanager
    async def mock_async_session():
        yield db_session

    with patch("polyflip.execution.worker.async_session", new=mock_async_session), \
         patch("polyflip.execution.worker.ExecutionSettings") as MockSettings, \
         patch("polyflip.execution.worker.check_risk_limits", return_value=None):
        MockSettings.return_value.execution_mode.value = "LIVE"
        await process_ready_requests()
        
    await db_session.refresh(req)
    await db_session.refresh(reservation)
    await db_session.refresh(trade)
    assert req.state == "REJECTED"
    assert req.error_reason and "NO_LIQUIDITY_FAK" in req.error_reason
    assert reservation.released_at is not None
    assert trade.position_status == "ENTRY_FAILED"



def test_live_mode_allows_missing_credentials_when_kill_switch_off(monkeypatch):
    from polyflip.execution.config import ExecutionSettings

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    for name in (
        "POLYGON_PRIVATE_KEY",
        "POLYGON_ADDRESS",
        "POLYMARKET_RELAYER_API_KEY",
        "POLYMARKET_RELAYER_API_KEY_ADDRESS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = ExecutionSettings()
    assert settings.live_trading_enabled is False
