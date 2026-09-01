from unittest.mock import patch, AsyncMock
"""
tests/test_release_gate.py — Тесты Этапа 8: release_gate

Инварианты, проверяемые здесь:
  1. release_gate создаёт TradeHistory(mode=LIVE) + ExecutionRequest(READY)
     из LiveMirrorCandidate(ELIGIBLE) при LIVE_RELEASE_MODE=AUTO
  2. Кандидат переходит в RELEASED атомарно в той же транзакции
  3. При LIVE_RELEASE_MODE=DISABLED ничего не выпускается
  4. При LIVE_RELEASE_MODE=MANUAL обрабатываются только ELIGIBLE (не NEW)
  5. PAPER-строки НЕ изменяются
  6. source_paper_trade_id и source_paper_request_id корректно заполнены
  7. idempotency: повторный вызов release_batch ничего не создаёт
     (кандидат уже RELEASED — не попадает в выборку)
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select, func

from polyflip.db.execution_models import (
    ExecutionRequest,
    LiveMirrorCandidate,
    ExposureReservation,
)
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.execution.release_gate import release_batch, _get_release_mode

# ─── Вспомогательные функции ──────────────────────────────────────────────────


async def _set_release_mode(session, mode: str) -> None:
    now = datetime.now(timezone.utc)
    existing = (
        await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_RELEASE_MODE")
        )
    ).scalar_one_or_none()
    if existing:
        existing.value = mode
        existing.updated_at = now
        existing.updated_by = "test"
    else:
        session.add(
            RuntimeSettings(
                key="LIVE_RELEASE_MODE", value=mode, updated_at=now, updated_by="test"
            )
        )
    await session.commit()


async def _make_paper_trade(session, market_id="MKT-TEST"):
    from polyflip.db.models import LiveMarket
    now = datetime.now(timezone.utc)
    # Ensure LiveMarket exists for this market_id
    existing_market = await session.scalar(select(LiveMarket).where(LiveMarket.market_id == market_id))
    if not existing_market:
        session.add(
            LiveMarket(
                market_id=market_id,
                asset="BTC",
                question="Test question",
                yes_token_id=f"token-yes-{market_id}",
                no_token_id=f"token-no-{market_id}",
                end_time_est=now + timedelta(hours=24),
                current_yes_price=0.5,
                current_no_price=0.5,
                current_spread=0.01,
                last_updated=now,
            )
        )
    t = TradeHistory(
        market_id=market_id,
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=5.0,
        executed_price=0.5,
        predicted_flip_prob=0.7,
        active_features="f1",
        status="SUCCESS",
        mode="PAPER",
        position_status="OPEN",
        entry_filled_shares=Decimal("10"),
        entry_cost_usdc=Decimal("5"),
        remaining_shares=Decimal("10"),
        realized_pnl_usdc=Decimal("0"),
        model_key="BTC_v3",
        model_version=3,
        edge=0.08,
        market_role="FAVORITE",
        created_at=datetime.now(timezone.utc),
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t


async def _make_paper_request(session, trade, amount_usdc: Decimal = Decimal("5")):
    req = ExecutionRequest(
        id=uuid.uuid4(),
        idempotency_key=f"PAPER-OPEN-{uuid.uuid4()}",
        requested_mode="PAPER",
        trade_history_id=trade.id,
        intent="OPEN",
        trigger_reason="STRATEGY",
        market_id=trade.market_id,
        asset=trade.asset,
        outcome_to_buy="YES",
        target_amount_usdc=amount_usdc,
        requested_shares=Decimal("10"),
        limit_price=Decimal("0.5"),
        max_slippage_pct=0.02,
        max_spend_usdc=amount_usdc,
        state="FILLED",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(req)
    await session.commit()
    await session.refresh(req)
    return req


from polyflip.execution.live_mirror_worker import _build_signal_snapshot, _compute_hash


async def _make_candidate(
    session, paper_request, paper_trade, state="ELIGIBLE", target_mode="SHADOW"
):
    snap = _build_signal_snapshot(paper_request, paper_trade)
    sig_hash = _compute_hash(snap)
    c = LiveMirrorCandidate(
        id=uuid.uuid4(),
        source_paper_request_id=paper_request.id,
        source_paper_trade_id=paper_trade.id,
        target_mode=target_mode,
        state=state,
        signal_snapshot=snap,
        signal_hash=sig_hash,
        created_at=datetime.now(timezone.utc),
    )
    session.add(c)
    await session.commit()
    await session.refresh(c)
    return c


# ─── Тесты ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_mode_disabled_releases_nothing(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """При LIVE_RELEASE_MODE=DISABLED ничего не выпускается."""
    await _set_release_mode(db_session, "DISABLED")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    released = await release_batch(db_session, "SHADOW")
    assert released == 0


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_batch_auto_releases_eligible(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """При AUTO release_gate выпускает ELIGIBLE кандидата."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    released = await release_batch(db_session, "SHADOW")
    assert released == 1

    await db_session.refresh(candidate)
    assert candidate.state == "RELEASED"
    assert candidate.released_at is not None
    assert candidate.released_trade_id is not None
    assert candidate.released_request_id is not None


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_batch_auto_releases_new(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """При AUTO release_gate выпускает и NEW-кандидатов."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(db_session, req, trade, state="NEW")

    released = await release_batch(db_session, "SHADOW")
    assert released == 1

    await db_session.refresh(candidate)
    assert candidate.state == "RELEASED"


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_batch_manual_skips_new(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """При MANUAL release_gate НЕ выпускает NEW-кандидатов."""
    await _set_release_mode(db_session, "MANUAL")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(db_session, req, trade, state="NEW")

    released = await release_batch(db_session, "SHADOW")
    assert released == 0

    await db_session.refresh(candidate)
    assert candidate.state == "NEW"  # не тронут


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_creates_live_trade_and_request(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """release_gate создаёт TradeHistory(mode=SHADOW) и ExecutionRequest(mode=SHADOW, state=READY)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    await release_batch(db_session, "SHADOW")

    await db_session.refresh(candidate)
    live_trade_id = candidate.released_trade_id
    live_req_id = candidate.released_request_id

    live_trade = await db_session.get(TradeHistory, live_trade_id)
    live_req = await db_session.get(ExecutionRequest, live_req_id)

    assert live_trade is not None
    assert live_trade.mode == "SHADOW"
    assert live_trade.position_status == "OPENING"
    assert live_trade.status == "PENDING"
    assert live_trade.source_paper_trade_id == trade.id

    assert live_req is not None
    assert live_req.requested_mode == "SHADOW"
    assert live_req.state == "READY"
    assert live_req.source_paper_request_id == req.id
    assert live_req.intent == "OPEN"
    assert live_req.trigger_reason == "MIRROR"


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_paper_rows_unchanged(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """PAPER-строки после release_batch не изменяются."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    paper_req_state_before = req.state
    paper_trade_status_before = trade.status

    await release_batch(db_session, "SHADOW")

    await db_session.refresh(trade)
    await db_session.refresh(req)

    assert trade.status == paper_trade_status_before
    assert trade.mode == "PAPER"
    assert trade.source_paper_trade_id is None  # PAPER не указывает сама на себя
    assert req.state == paper_req_state_before
    assert req.requested_mode == "PAPER"


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_batch_idempotent(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """Повторный release_batch не создаёт дублей (RELEASED не попадает в выборку)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    first = await release_batch(db_session, "SHADOW")
    second = await release_batch(db_session, "SHADOW")

    assert first == 1
    assert second == 0

    live_trades = (
        await db_session.scalars(
            select(TradeHistory).where(TradeHistory.mode == "SHADOW")
        )
    ).all()
    assert len(live_trades) == 1


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_no_new_paper_rows_created(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """release_batch не создаёт PAPER-строк."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    await release_batch(db_session, "SHADOW")

    paper_trades = (
        await db_session.scalars(
            select(TradeHistory).where(TradeHistory.mode == "PAPER")
        )
    ).all()
    paper_reqs = (
        await db_session.scalars(
            select(ExecutionRequest).where(ExecutionRequest.requested_mode == "PAPER")
        )
    ).all()

    assert len(paper_trades) == 1  # только оригинальный
    assert len(paper_reqs) == 1  # только оригинальный


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_get_release_mode_default_disabled(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """Без записи в RuntimeSettings release mode = DISABLED."""
    mode = await _get_release_mode(db_session)
    assert mode == "DISABLED"


@pytest.mark.asyncio
async def test_release_deferred_keeps_candidate_eligible_when_kill_switch_off(
    db_session,
):
    """Когда kill switch выключен в LIVE режиме, кандидат откладывается (остаётся ELIGIBLE, не REJECTED)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(
        db_session, req, trade, state="ELIGIBLE", target_mode="LIVE"
    )

    released = await release_batch(db_session, "LIVE")
    assert released == 0

    await db_session.refresh(candidate)
    assert candidate.state == "ELIGIBLE"  # Не переведён в REJECTED!


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_rejected_when_signal_too_old(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """Когда PAPER-сигнал старше 30 секунд, кандидат забраковывается (state=REJECTED)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    db_session.add(RuntimeSettings(
        key="LIVE_MAX_SIGNAL_AGE_SEC", value="30",
        updated_at=datetime.now(timezone.utc), updated_by="test",
    ))


    # Старый сигнал (40 секунд назад)
    req.updated_at = datetime.now(timezone.utc) - timedelta(seconds=40)
    req.created_at = req.updated_at
    await db_session.commit()

    candidate = await _make_candidate(
        db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW"
    )

    released = await release_batch(db_session, "SHADOW")
    assert released == 0

    await db_session.refresh(candidate)
    assert candidate.state == "REJECTED"
    assert "Signal is too old" in candidate.rejection_reason


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_sets_expires_at_and_ttl(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """Создаваемая LIVE-заявка обязана иметь ttl_seconds <= 30 и заполненный expires_at."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    req.ttl_seconds = 120  # должно усечься до 30
    await db_session.commit()

    candidate = await _make_candidate(
        db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW"
    )

    await release_batch(db_session, "SHADOW")

    await db_session.refresh(candidate)
    live_req = await db_session.get(ExecutionRequest, candidate.released_request_id)
    assert live_req is not None
    assert live_req.ttl_seconds == 30
    assert live_req.expires_at is not None


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_release_creates_exposure_reservation(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """При выпуске кандидата атомарно создаётся ExposureReservation."""
    from polyflip.db.execution_models import ExposureReservation

    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(
        db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW"
    )

    released = await release_batch(db_session, "SHADOW")
    assert released == 1

    await db_session.refresh(candidate)
    reservation = await db_session.scalar(
        select(ExposureReservation).where(
            ExposureReservation.request_id == candidate.released_request_id
        )
    )

    assert reservation is not None
    assert reservation.request_id == candidate.released_request_id
    assert reservation.trade_history_id == candidate.released_trade_id
    assert reservation.amount_usdc == Decimal("5")
    assert reservation.released_at is None


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_concurrent_release_respects_total_exposure(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """
    Проверяет, что при установленном лимите MAX_TOTAL_EXPOSURE_USDC=1.0
    повторный выпуск кандидатов блокируется лимитом экспозиции.
    """
    from polyflip.execution.risk_checks import check_risk_limits

    now = datetime.now(timezone.utc)
    db_session.add(
        RuntimeSettings(
            key="MAX_TOTAL_EXPOSURE_USDC",
            value="1.0",
            updated_at=now,
            updated_by="test",
        )
    )
    db_session.add(
        RuntimeSettings(
            key="SHADOW_RISK_LIMITS_ENABLED",
            value="true",
            updated_at=now,
            updated_by="test",
        )
    )
    await db_session.commit()

    err1 = await check_risk_limits(
        db_session,
        intent="OPEN",
        max_spend_usdc=Decimal("1.0"),
        requested_mode="SHADOW",
    )
    assert err1 is None

    # Занимаем лимит
    t = await _make_paper_trade(db_session)
    t.mode = "SHADOW"
    t.entry_cost_usdc = Decimal("1.0")
    t.entry_filled_shares = Decimal("2.0")
    t.remaining_shares = Decimal("2.0")
    await db_session.commit()

    err2 = await check_risk_limits(
        db_session,
        intent="OPEN",
        max_spend_usdc=Decimal("1.0"),
        requested_mode="SHADOW",
    )
    assert err2 is not None
    assert "Max total exposure" in err2


@pytest.mark.postgres
@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_two_release_gates_cannot_exceed_total_exposure(mock_client_class, pg_session_factory):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """
    PostgreSQL тест: гарантирует, что 2 конкурентных Release Gate не превысят лимит экспозиции.
    """
    import asyncio
    from polyflip.execution.release_gate import release_candidate_by_id

    async with pg_session_factory() as setup_session:
        now = datetime.now(timezone.utc)
        setup_session.add(
            RuntimeSettings(
                key="MAX_TOTAL_EXPOSURE_USDC",
                value="5.0",
                updated_at=now,
                updated_by="test",
            )
        )
        setup_session.add(
            RuntimeSettings(
                key="SHADOW_RISK_LIMITS_ENABLED",
                value="true",
                updated_at=now,
                updated_by="test",
            )
        )

        p_trade1 = await _make_paper_trade(setup_session, market_id="MKT-PG-1")
        p_req1 = await _make_paper_request(
            setup_session, p_trade1, amount_usdc=Decimal("4.0")
        )

        p_trade2 = await _make_paper_trade(setup_session, market_id="MKT-PG-2")
        p_req2 = await _make_paper_request(
            setup_session, p_trade2, amount_usdc=Decimal("4.0")
        )

        cand1 = await _make_candidate(setup_session, p_req1, p_trade1)
        cand2 = await _make_candidate(setup_session, p_req2, p_trade2)
        cand1.target_mode = "SHADOW"
        cand2.target_mode = "SHADOW"

        await setup_session.commit()
        cand1_id = cand1.id
        cand2_id = cand2.id

    async with pg_session_factory() as s1, pg_session_factory() as s2:
        results = await asyncio.gather(
            release_candidate_by_id(s1, cand1_id, "SHADOW"),
            release_candidate_by_id(s2, cand2_id, "SHADOW"),
            return_exceptions=True,
        )

    assert results.count(True) == 1
    assert results.count(False) == 1
    assert not any(isinstance(result, BaseException) for result in results)

    candidate_ids = [cand1_id, cand2_id]
    paper_request_ids = [p_req1.id, p_req2.id]

    async with pg_session_factory() as check_session:
        released_candidates = await check_session.scalar(
            select(func.count())
            .select_from(LiveMirrorCandidate)
            .where(
                LiveMirrorCandidate.id.in_(candidate_ids),
                LiveMirrorCandidate.state == "RELEASED",
            )
        )

        live_request_ids = (
            await check_session.scalars(
                select(ExecutionRequest.id).where(
                    ExecutionRequest.requested_mode == "SHADOW",
                    ExecutionRequest.source_paper_request_id.in_(paper_request_ids),
                )
            )
        ).all()

        res_sum = await check_session.scalar(
            select(
                func.coalesce(
                    func.sum(ExposureReservation.amount_usdc),
                    Decimal("0"),
                )
            ).where(
                ExposureReservation.request_id.in_(live_request_ids),
                ExposureReservation.released_at.is_(None),
            )
        )

    assert released_candidates == 1
    assert len(live_request_ids) == 1
    assert res_sum == Decimal("4.0")

@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_one_dollar_paper_becomes_1_10_live(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    """PAPER заявка на 1.00 USDC превращается в LIVE-заявку на 1.10 USDC."""
    from polyflip.execution.release_gate import release_candidate_by_id
    from polyflip.db.execution_models import LiveTradingSession, ExposureReservation, ExecutionWorkerStatus
    from polyflip.db.models import RuntimeSettings
    import uuid

    # Настраиваем окружение
    now = datetime.now(timezone.utc)
    db_session.add(
        RuntimeSettings(
            key="LIVE_TRADING_ENABLED",
            value="true",
            updated_at=now,
            updated_by="test",
        )
    )

    # Добавляем LIVE worker status
    db_session.add(
        ExecutionWorkerStatus(
            worker_id=str(uuid.uuid4()),
            execution_mode="LIVE",
            heartbeat_at=now,
            gateway_ready=True,
            collateral_allowance_ready=True,
            conditional_allowance_ready=True,
            balance_usdc=Decimal("100.0"),
        )
    )

    # Создаём сессию
    session_obj = LiveTradingSession(
        status="ACTIVE",
        budget_usdc=Decimal("10.0"),
        max_single_order_usdc=Decimal("1.10"),
        max_open_positions=5,
        max_total_exposure_usdc=Decimal("10.0"),
        reserved_usdc=Decimal("0"),
        filled_usdc=Decimal("0"),
    )
    db_session.add(session_obj)
    await db_session.commit()

    # PAPER-заявка и сделка
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade, amount_usdc=Decimal("1.00"))

    # обновляем таймстемпы, чтобы не было ошибки Signal is too old
    trade.created_at = now
    req.created_at = now
    req.updated_at = now
    await db_session.commit()

    candidate = await _make_candidate(
        db_session, req, trade, state="ELIGIBLE", target_mode="LIVE"
    )

    # Релизим
    success = await release_candidate_by_id(db_session, candidate.id, "LIVE")
    assert success is True

    await db_session.refresh(candidate)
    assert candidate.state == "RELEASED"

    # Проверяем LIVE-заявку и резерв
    live_req = await db_session.get(ExecutionRequest, candidate.released_request_id)
    live_trade = await db_session.get(TradeHistory, candidate.released_trade_id)
    reservation = await db_session.scalar(
        select(ExposureReservation).where(
            ExposureReservation.request_id == live_req.id
        )
    )

    # Проверки по заданию
    assert Decimal(str(round(live_req.target_amount_usdc, 2))) == Decimal("1.10")
    assert Decimal(str(round(live_req.max_spend_usdc, 2))) == Decimal("1.10")
    assert Decimal(str(round(live_trade.amount_usdc, 2))) == Decimal("1.10")
    assert Decimal(str(round(reservation.amount_usdc, 2))) == Decimal("1.10")

    # PAPER не изменён
    await db_session.refresh(req)
    assert req.target_amount_usdc == Decimal("1.00")


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_quote_unavailable_is_deferred(mock_client_class, db_session):
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    from polyflip.execution.release_gate import release_candidate_by_id
    from polyflip.db.models import RuntimeSettings, LiveMarket

    now = datetime.now(timezone.utc)
    
    # Создаём маркет, чтобы не упало на "Market MKT-TEST not found"
    db_session.add(LiveMarket(
        market_id="MKT-TEST", 
        asset="BTC", 
        yes_token_id="T1", 
        no_token_id="T2", 
        question="Q", 
        end_time_est=now,
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.0,
        volume_5min=0.0,
        price_velocity=0.0,
        accepting_orders=True,
        trading_status="active",
        resolution_status="unresolved",
        last_updated=now,
    ))
    
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade, amount_usdc=Decimal("1.00"))

    trade.created_at = now
    req.created_at = now
    req.updated_at = now
    await db_session.commit()

    candidate = await _make_candidate(
        db_session, req, trade, state="ELIGIBLE", target_mode="LIVE"
    )

    # Имитируем отсутствие котировки
    mock_api_client = AsyncMock()
    mock_api_client.get_market_prices.return_value = {}

    success = await release_candidate_by_id(db_session, candidate.id, "LIVE", api_client=mock_api_client)
    
    assert success is False
    
    # Кандидат не должен быть отклонен (REJECTED), он остается NEW/ELIGIBLE
    await db_session.refresh(candidate)
    assert candidate.state != "REJECTED"


@pytest.mark.asyncio
@patch('polyflip.collector.client.PolymarketClient')
async def test_shadow_release_allows_one_dollar_source_trade(mock_client_class, db_session):
    """SHADOW mirrors must not inherit the LIVE $1.10 session minimum."""
    mock_client = mock_client_class.return_value
    mock_client.get_market_prices = AsyncMock(return_value={"best_ask": 0.5, "best_bid": 0.5})
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session, market_id="MKT-SHADOW-ONE-DOLLAR")
    req = await _make_paper_request(db_session, trade, amount_usdc=Decimal("1.00"))
    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW")

    released = await release_batch(db_session, "SHADOW")

    assert released == 1
    await db_session.refresh(candidate)
    assert candidate.state == "RELEASED"
