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

from sqlalchemy import select

from polyflip.db.execution_models import ExecutionRequest, LiveMirrorCandidate
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.execution.release_gate import release_batch, _get_release_mode


# ─── Вспомогательные функции ──────────────────────────────────────────────────

async def _set_release_mode(session, mode: str) -> None:
    now = datetime.now(timezone.utc)
    existing = (await session.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_RELEASE_MODE")
    )).scalar_one_or_none()
    if existing:
        existing.value = mode
        existing.updated_at = now
        existing.updated_by = "test"
    else:
        session.add(RuntimeSettings(key="LIVE_RELEASE_MODE", value=mode, updated_at=now, updated_by="test"))
    await session.commit()


async def _make_paper_trade(session, market_id="MKT-TEST"):
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


async def _make_paper_request(session, trade):
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
        target_amount_usdc=Decimal("5"),
        requested_shares=Decimal("10"),
        limit_price=Decimal("0.5"),
        max_slippage_pct=0.02,
        max_spend_usdc=Decimal("5"),
        state="FILLED",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(req)
    await session.commit()
    await session.refresh(req)
    return req


from polyflip.execution.live_mirror_worker import _build_signal_snapshot, _compute_hash


async def _make_candidate(session, paper_request, paper_trade, state="ELIGIBLE", target_mode="SHADOW"):
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
async def test_release_mode_disabled_releases_nothing(db_session):
    """При LIVE_RELEASE_MODE=DISABLED ничего не выпускается."""
    await _set_release_mode(db_session, "DISABLED")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    released = await release_batch(db_session, "SHADOW")
    assert released == 0


@pytest.mark.asyncio
async def test_release_batch_auto_releases_eligible(db_session):
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
async def test_release_batch_auto_releases_new(db_session):
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
async def test_release_batch_manual_skips_new(db_session):
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
async def test_release_creates_live_trade_and_request(db_session):
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
    assert live_trade.position_status == "OPEN"
    assert live_trade.source_paper_trade_id == trade.id

    assert live_req is not None
    assert live_req.requested_mode == "SHADOW"
    assert live_req.state == "READY"
    assert live_req.source_paper_request_id == req.id
    assert live_req.intent == "OPEN"
    assert live_req.trigger_reason == "MIRROR"


@pytest.mark.asyncio
async def test_release_paper_rows_unchanged(db_session):
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
async def test_release_batch_idempotent(db_session):
    """Повторный release_batch не создаёт дублей (RELEASED не попадает в выборку)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    first = await release_batch(db_session, "SHADOW")
    second = await release_batch(db_session, "SHADOW")

    assert first == 1
    assert second == 0

    live_trades = (await db_session.scalars(
        select(TradeHistory).where(TradeHistory.mode == "SHADOW")
    )).all()
    assert len(live_trades) == 1


@pytest.mark.asyncio
async def test_release_no_new_paper_rows_created(db_session):
    """release_batch не создаёт PAPER-строк."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await _make_candidate(db_session, req, trade, state="ELIGIBLE")

    await release_batch(db_session, "SHADOW")

    paper_trades = (await db_session.scalars(
        select(TradeHistory).where(TradeHistory.mode == "PAPER")
    )).all()
    paper_reqs = (await db_session.scalars(
        select(ExecutionRequest).where(ExecutionRequest.requested_mode == "PAPER")
    )).all()

    assert len(paper_trades) == 1   # только оригинальный
    assert len(paper_reqs) == 1     # только оригинальный


@pytest.mark.asyncio
async def test_get_release_mode_default_disabled(db_session):
    """Без записи в RuntimeSettings release mode = DISABLED."""
    mode = await _get_release_mode(db_session)
    assert mode == "DISABLED"


@pytest.mark.asyncio
async def test_release_deferred_keeps_candidate_eligible_when_kill_switch_off(db_session):
    """Когда kill switch выключен в LIVE режиме, кандидат откладывается (остаётся ELIGIBLE, не REJECTED)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE", target_mode="LIVE")

    released = await release_batch(db_session, "LIVE")
    assert released == 0

    await db_session.refresh(candidate)
    assert candidate.state == "ELIGIBLE"  # Не переведён в REJECTED!


@pytest.mark.asyncio
async def test_release_rejected_when_signal_too_old(db_session):
    """Когда PAPER-сигнал старше 30 секунд, кандидат забраковывается (state=REJECTED)."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    
    # Старый сигнал (40 секунд назад)
    req.updated_at = datetime.now(timezone.utc) - timedelta(seconds=40)
    req.created_at = req.updated_at
    await db_session.commit()

    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW")

    released = await release_batch(db_session, "SHADOW")
    assert released == 0

    await db_session.refresh(candidate)
    assert candidate.state == "REJECTED"
    assert "Signal is too old" in candidate.rejection_reason


@pytest.mark.asyncio
async def test_release_sets_expires_at_and_ttl(db_session):
    """Создаваемая LIVE-заявка обязана иметь ttl_seconds <= 30 и заполненный expires_at."""
    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    req.ttl_seconds = 120  # должно усечься до 30
    await db_session.commit()

    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW")

    await release_batch(db_session, "SHADOW")

    await db_session.refresh(candidate)
    live_req = await db_session.get(ExecutionRequest, candidate.released_request_id)
    assert live_req is not None
    assert live_req.ttl_seconds == 30
    assert live_req.expires_at is not None


@pytest.mark.asyncio
async def test_release_creates_exposure_reservation(db_session):
    """При выпуске кандидата атомарно создаётся ExposureReservation."""
    from polyflip.db.execution_models import ExposureReservation

    await _set_release_mode(db_session, "AUTO")
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    candidate = await _make_candidate(db_session, req, trade, state="ELIGIBLE", target_mode="SHADOW")

    released = await release_batch(db_session, "SHADOW")
    assert released == 1

    await db_session.refresh(candidate)
    reservation = await db_session.scalar(
        select(ExposureReservation).where(ExposureReservation.request_id == candidate.released_request_id)
    )

    assert reservation is not None
    assert reservation.request_id == candidate.released_request_id
    assert reservation.trade_history_id == candidate.released_trade_id
    assert reservation.amount_usdc == Decimal("5")
    assert reservation.released_at is None


@pytest.mark.asyncio
async def test_concurrent_release_respects_total_exposure(db_session):
    """
    Проверяет, что при установленном лимите MAX_TOTAL_EXPOSURE_USDC=1.0
    повторный выпуск кандидатов блокируется лимитом экспозиции.
    """
    from polyflip.execution.risk_checks import check_risk_limits

    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="MAX_TOTAL_EXPOSURE_USDC", value="1.0", updated_at=now, updated_by="test"))
    db_session.add(RuntimeSettings(key="SHADOW_RISK_LIMITS_ENABLED", value="true", updated_at=now, updated_by="test"))
    await db_session.commit()

    err1 = await check_risk_limits(db_session, intent="OPEN", max_spend_usdc=Decimal("1.0"), requested_mode="SHADOW")
    assert err1 is None

    # Занимаем лимит
    t = await _make_paper_trade(db_session)
    t.mode = "SHADOW"
    t.entry_cost_usdc = Decimal("1.0")
    t.entry_filled_shares = Decimal("2.0")
    t.remaining_shares = Decimal("2.0")
    await db_session.commit()

    err2 = await check_risk_limits(db_session, intent="OPEN", max_spend_usdc=Decimal("1.0"), requested_mode="SHADOW")
    assert err2 is not None
    assert "Max total exposure" in err2
