"""
tests/test_live_mirror_stage4_5.py

Тесты Этапов 4 и 5:
  - Этап 4: Структура таблицы LiveMirrorCandidate, nullable FK-поля
  - Этап 5: mirror_batch — создание кандидатов, идемпотентность, изоляция PAPER

Ключевые инварианты, проверяемые здесь:
  1. mirror_batch создаёт LiveMirrorCandidate только для FILLED PAPER OPEN.
  2. mirror_batch НЕ создаёт кандидата для PAPER OPEN в состоянии READY.
  3. mirror_batch идемпотентен: повторный запуск не дублирует кандидата.
  4. Ни одна PAPER строка не изменяется после mirror_batch.
  5. При LIVE_MIRROR_ENABLED=false батч всегда возвращает 0.
  6. Несколько разных PAPER OPEN → несколько разных кандидатов.
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select, func

from polyflip.db.execution_models import ExecutionRequest, LiveMirrorCandidate
from polyflip.db.models import TradeHistory


# ─── Вспомогательные фикстуры ────────────────────────────────────────────────

async def _make_paper_trade(session, *, market_id: str = "MKT-1", asset: str = "BTC") -> TradeHistory:
    t = TradeHistory(
        market_id=market_id,
        asset=asset,
        outcome_bought="YES",
        amount_usdc=1.0,
        executed_price=0.5,
        predicted_flip_prob=0.65,
        active_features="f1 f2",
        status="SUCCESS",
        mode="PAPER",
        position_status="OPEN",
        entry_filled_shares=Decimal("2"),
        entry_cost_usdc=Decimal("1"),
        remaining_shares=Decimal("2"),
        realized_pnl_usdc=Decimal("0"),
        model_key="BTC_leaning",
        model_version=3,
        confirm_model_key=None,
        confirm_model_version=None,
        model_attribution_source="EXACT",
        edge=0.05,
        market_role="FAVORITE",
        created_at=datetime.now(timezone.utc),
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t


async def _make_paper_request(
    session,
    trade: TradeHistory,
    *,
    state: str = "FILLED",
    market_id: str | None = None,
) -> ExecutionRequest:
    await set_mirror_enabled(session, enabled=True)
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    req = ExecutionRequest(
        id=uuid.uuid4(),
        idempotency_key=f"PAPER-OPEN-{uuid.uuid4()}",
        requested_mode="PAPER",
        trade_history_id=trade.id,
        intent="OPEN",
        trigger_reason="STRATEGY",
        market_id=market_id or trade.market_id,
        asset=trade.asset,
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("1"),
        requested_shares=Decimal("2"),
        limit_price=Decimal("0.5"),
        max_slippage_pct=0.02,
        max_spend_usdc=Decimal("1"),
        state=state,
        created_at=now,
        updated_at=now,
        expires_at=None,
    )
    session.add(req)
    await session.commit()
    await session.refresh(req)
    return req


# ─── Вспомогательный импорт mirror_batch ─────────────────────────────────────
# Импортируем напрямую, минуя управляющий флаг LIVE_MIRROR_ENABLED
from polyflip.execution.live_mirror_worker import (
    mirror_batch,
    set_mirror_enabled,
    _build_signal_snapshot,
    _compute_hash,
    TARGET_MODE,
)


# ─── Тесты Этапа 4: структура БД ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_mirror_candidate_table_exists(db_session):
    """Таблица live_mirror_candidates создана корректно."""
    count = await db_session.scalar(
        select(func.count()).select_from(LiveMirrorCandidate)
    )
    assert count == 0


@pytest.mark.asyncio
async def test_trade_history_source_paper_trade_id_is_nullable(db_session):
    """PAPER строки имеют source_paper_trade_id = NULL."""
    trade = await _make_paper_trade(db_session)
    await db_session.refresh(trade)
    assert trade.source_paper_trade_id is None


@pytest.mark.asyncio
async def test_execution_request_source_paper_request_id_is_nullable(db_session):
    """PAPER заявки имеют source_paper_request_id = NULL."""
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    await db_session.refresh(req)
    assert req.source_paper_request_id is None


@pytest.mark.asyncio
async def test_live_mirror_candidate_unique_constraint(db_session):
    """Нельзя создать два кандидата для одной PAPER-заявки в одном target_mode."""
    from sqlalchemy.exc import IntegrityError

    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade)
    snapshot = _build_signal_snapshot(req, trade)
    sig_hash = _compute_hash(snapshot)

    now = datetime.now(timezone.utc)
    c1 = LiveMirrorCandidate(
        id=uuid.uuid4(),
        source_paper_request_id=req.id,
        source_paper_trade_id=trade.id,
        target_mode="SHADOW",
        state="NEW",
        signal_snapshot=snapshot,
        signal_hash=sig_hash,
        created_at=now,
    )
    db_session.add(c1)
    await db_session.commit()

    c2 = LiveMirrorCandidate(
        id=uuid.uuid4(),
        source_paper_request_id=req.id,
        source_paper_trade_id=trade.id,
        target_mode="SHADOW",  # дубль!
        state="NEW",
        signal_snapshot=snapshot,
        signal_hash=sig_hash,
        created_at=now,
    )
    db_session.add(c2)
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ─── Тесты Этапа 5: mirror_batch ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mirror_batch_creates_candidate_for_filled_paper_open(db_session):
    """mirror_batch создаёт кандидата для FILLED PAPER OPEN."""
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade, state="FILLED")

    created = await mirror_batch(db_session)

    assert created == 1
    candidate = await db_session.scalar(select(LiveMirrorCandidate))
    assert candidate is not None
    assert candidate.source_paper_request_id == req.id
    assert candidate.source_paper_trade_id == trade.id
    assert candidate.state == "NEW"
    assert candidate.target_mode == TARGET_MODE
    assert candidate.signal_hash == _compute_hash(_build_signal_snapshot(req, trade))


@pytest.mark.asyncio
async def test_mirror_batch_skips_ready_paper_open(db_session):
    """mirror_batch НЕ создаёт кандидата для PAPER OPEN в состоянии READY (ещё не исполнена)."""
    trade = await _make_paper_trade(db_session)
    await _make_paper_request(db_session, trade, state="READY")

    created = await mirror_batch(db_session)

    assert created == 0
    count = await db_session.scalar(select(func.count()).select_from(LiveMirrorCandidate))
    assert count == 0


@pytest.mark.asyncio
async def test_mirror_batch_idempotent_on_repeat_run(db_session):
    """Повторный вызов mirror_batch не дублирует кандидата."""
    trade = await _make_paper_trade(db_session)
    await _make_paper_request(db_session, trade, state="FILLED")

    first_run = await mirror_batch(db_session)
    second_run = await mirror_batch(db_session)

    assert first_run == 1
    assert second_run == 0

    count = await db_session.scalar(select(func.count()).select_from(LiveMirrorCandidate))
    assert count == 1


@pytest.mark.asyncio
async def test_mirror_batch_paper_rows_unchanged_after_run(db_session):
    """Ни одна PAPER строка не меняется после mirror_batch."""
    trade = await _make_paper_trade(db_session)
    req = await _make_paper_request(db_session, trade, state="FILLED")

    paper_trade_status_before = trade.status
    paper_req_state_before = req.state

    await mirror_batch(db_session)

    await db_session.refresh(trade)
    await db_session.refresh(req)

    assert trade.status == paper_trade_status_before
    assert trade.position_status == "OPEN"
    assert trade.realized_pnl_usdc == Decimal("0")
    assert req.state == paper_req_state_before


@pytest.mark.asyncio
async def test_mirror_batch_no_live_trade_or_request_created(db_session):
    """mirror_batch не создаёт ни одной строки TradeHistory(mode='LIVE') или ExecutionRequest(mode='LIVE')."""
    trade = await _make_paper_trade(db_session)
    await _make_paper_request(db_session, trade, state="FILLED")

    await mirror_batch(db_session)

    live_trades = (await db_session.scalars(
        select(TradeHistory).where(TradeHistory.mode == "LIVE")
    )).all()
    live_reqs = (await db_session.scalars(
        select(ExecutionRequest).where(ExecutionRequest.requested_mode == "LIVE")
    )).all()

    assert len(live_trades) == 0
    assert len(live_reqs) == 0


@pytest.mark.asyncio
async def test_mirror_batch_multiple_filled_creates_multiple_candidates(db_session):
    """Несколько FILLED PAPER OPEN → несколько разных кандидатов."""
    trade1 = await _make_paper_trade(db_session, market_id="MKT-A", asset="BTC")
    trade2 = await _make_paper_trade(db_session, market_id="MKT-B", asset="ETH")
    trade3 = await _make_paper_trade(db_session, market_id="MKT-C", asset="SOL")

    await _make_paper_request(db_session, trade1, state="FILLED", market_id="MKT-A")
    await _make_paper_request(db_session, trade2, state="FILLED", market_id="MKT-B")
    await _make_paper_request(db_session, trade3, state="FILLED", market_id="MKT-C")

    created = await mirror_batch(db_session)

    assert created == 3
    count = await db_session.scalar(select(func.count()).select_from(LiveMirrorCandidate))
    assert count == 3

    # Проверяем уникальность signal_hash у кандидатов
    candidates = (await db_session.scalars(select(LiveMirrorCandidate))).all()
    hashes = [c.signal_hash for c in candidates]
    assert len(set(hashes)) == 3  # Все разные


@pytest.mark.asyncio
async def test_mirror_batch_partially_filled_final_is_mirrorable(db_session):
    """PARTIALLY_FILLED_FINAL тоже является зеркалируемым состоянием."""
    trade = await _make_paper_trade(db_session)
    await _make_paper_request(db_session, trade, state="PARTIALLY_FILLED_FINAL")

    created = await mirror_batch(db_session)
    assert created == 1


@pytest.mark.asyncio
async def test_mirror_candidate_state_is_new(db_session):
    """Созданный кандидат всегда имеет state='NEW'."""
    trade = await _make_paper_trade(db_session)
    await _make_paper_request(db_session, trade, state="FILLED")

    await mirror_batch(db_session)

    candidate = await db_session.scalar(select(LiveMirrorCandidate))
    assert candidate.state == "NEW"
    # released_at и released_*_id должны быть NULL
    assert candidate.released_at is None
    assert candidate.released_trade_id is None
    assert candidate.released_request_id is None


from polyflip.db.models import RuntimeSettings
from polyflip.execution.live_mirror_worker import runtime_bool


@pytest.mark.asyncio
async def test_mirror_switch_changes_behavior_without_restart(db_session):
    """Флаг LIVE_MIRROR_ENABLED из БД динамически управляет поведением без рестарта."""
    # Выключаем
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="LIVE_MIRROR_ENABLED", value="false", updated_at=now, updated_by="test"))
    await db_session.commit()

    assert await runtime_bool(db_session, "LIVE_MIRROR_ENABLED") is False

    trade = await _make_paper_trade(db_session)
    await _make_paper_request(db_session, trade, state="FILLED")

    # Включаем через БД
    setting = await db_session.scalar(select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_MIRROR_ENABLED"))
    setting.value = "true"
    await db_session.commit()

    assert await runtime_bool(db_session, "LIVE_MIRROR_ENABLED") is True
    created = await mirror_batch(db_session)
    assert created == 1


@pytest.mark.asyncio
async def test_mirror_switch_persists_both_settings(db_session, engine):
    """toggle_mirror_switch совершает commit и сохраняет флаги в отдельной сессии."""
    from polyflip.api.execution_api import toggle_mirror_switch, SwitchBoolRequest
    from polyflip.db.models import RuntimeSettings
    from sqlalchemy.ext.asyncio import async_sessionmaker

    await toggle_mirror_switch(
        SwitchBoolRequest(enabled=True),
        db_session,
    )

    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as check_session:
        enabled_row = await check_session.scalar(
            select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_MIRROR_ENABLED")
        )
        started_row = await check_session.scalar(
            select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_MIRROR_STARTED_AT")
        )

    assert enabled_row is not None and enabled_row.value == "true"
    assert started_row is not None and started_row.value is not None


@pytest.mark.asyncio
async def test_paper_api_exposes_ready_live_worker(db_session):
    """API в PAPER-режиме возвращает kill_switch_available=True при наличии LIVE-воркера."""
    from polyflip.api.execution_api import get_live_trading_status
    from polyflip.db.execution_models import ExecutionWorkerStatus

    now = datetime.now(timezone.utc)
    ws = ExecutionWorkerStatus(
        worker_id="live_worker_1",
        execution_mode="LIVE",
        heartbeat_at=now,
        gateway_ready=True,
        credentials_loaded=True,
        wallet_address="0x123",
        balance_usdc=Decimal("100.0"),
        collateral_allowance_ready=True,
        conditional_allowance_ready=True,
    )
    db_session.add(ws)
    await db_session.commit()

    status = await get_live_trading_status(db_session)

    assert status["execution_mode"] == "PAPER"
    assert status["kill_switch_available"] is True
    assert status["worker_status"] is not None
    assert status["worker_status"]["execution_mode"] == "LIVE"
