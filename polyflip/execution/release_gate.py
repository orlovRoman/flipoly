"""
release_gate.py — Этап 8: шлюз выпуска LiveMirrorCandidate → LIVE TradeHistory + LIVE ExecutionRequest

Архитектура:
    mirror_worker                 release_gate              execution_worker_live
    ──────────────────────        ──────────────────────    ──────────────────────
    PAPER OPEN FILLED ──►         candidate(NEW/ELIGIBLE)   picks up LIVE OPEN READY
    LiveMirrorCandidate(NEW)      ──► atomically creates   ──► sends order to exchange
                                  TradeHistory(LIVE)
                                  ExecutionRequest(LIVE,READY)
                                  candidate.state=RELEASED

Инварианты (принудительные):
    - Никогда не изменяет PAPER TradeHistory или PAPER ExecutionRequest.
    - Атомарная транзакция: либо всё создаётся, либо ничего.
    - Кандидат переходит в RELEASED только внутри той же транзакции.
    - Если release_gate падает после INSERT TradeHistory, но до RELEASED —
      при следующем запуске он не увидит кандидата как RELEASED
      и попытается снова. Уникальный индекс uq_live_trade_source_paper
      заблокирует дубль INSERT → кандидат остаётся ELIGIBLE, оператор
      получает ошибку в логах.

Управление (читается из RuntimeSettings в БД, не из env):
    LIVE_RELEASE_MODE:
        DISABLED  — не делаем ничего (спим)
        MANUAL    — обрабатываем только кандидатов в состоянии ELIGIBLE
                    (оператор выставил через /api/execution/candidates/{id}/release)
        AUTO      — обрабатываем кандидатов в состоянии NEW и ELIGIBLE
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from polyflip.db.execution_models import ExecutionRequest, LiveMirrorCandidate
from polyflip.db.models import RuntimeSettings, TradeHistory

logger = logging.getLogger("release_gate")

# ── Настройки ────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
POLL_INTERVAL: float = float(os.environ.get("RELEASE_GATE_POLL_INTERVAL", "5"))

_shutdown = False


def _handle_sigterm(signum: int, frame: object) -> None:
    global _shutdown
    logger.info("Получен сигнал %s, остановка release_gate...", signum)
    _shutdown = True


# ── Вспомогательные функции ───────────────────────────────────────────────────

async def _get_release_mode(session: AsyncSession) -> str:
    """Читает LIVE_RELEASE_MODE из RuntimeSettings (DISABLED | MANUAL | AUTO)."""
    row = (await session.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_RELEASE_MODE")
    )).scalar_one_or_none()
    return (row.value if row else "DISABLED").upper()


async def _get_eligible_candidates(
    session: AsyncSession,
    release_mode: str,
    target_mode: str,
    batch_size: int = 10,
) -> list[LiveMirrorCandidate]:
    """
    Возвращает кандидатов, готовых к выпуску, в зависимости от режима:
      AUTO   — NEW и ELIGIBLE
      MANUAL — только ELIGIBLE
      DISABLED — []
    """
    if release_mode == "DISABLED":
        return []

    eligible_states = ["ELIGIBLE"] if release_mode == "MANUAL" else ["NEW", "ELIGIBLE"]

    stmt = (
        select(LiveMirrorCandidate)
        .where(
            LiveMirrorCandidate.state.in_(eligible_states),
            LiveMirrorCandidate.target_mode == target_mode,
        )
        .order_by(LiveMirrorCandidate.created_at)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )
    return (await session.execute(stmt)).scalars().all()


def _build_live_trade(
    candidate: LiveMirrorCandidate,
    paper_trade: TradeHistory,
    now: datetime,
    target_mode: str,
) -> TradeHistory:
    """
    Создаёт новый TradeHistory(mode=target_mode) на основе PAPER-снимка.

    Инварианты:
    - mode = target_mode (SHADOW или LIVE)
    - position_accounting_version = 0 (будет инициализировано после fill)
    - pnl / realized_pnl_usdc = None (неизвестно до исполнения)
    - source_paper_trade_id = paper_trade.id (ссылка на источник)
    """
    snap = candidate.signal_snapshot

    return TradeHistory(
        market_id=paper_trade.market_id,
        asset=paper_trade.asset,
        outcome_bought=paper_trade.outcome_bought,
        amount_usdc=paper_trade.amount_usdc,
        executed_price=0.0,               # заполняется после fill
        predicted_flip_prob=paper_trade.predicted_flip_prob,
        active_features=paper_trade.active_features,
        model_version=paper_trade.model_version,
        status="SUCCESS",
        mode=target_mode,
        edge=paper_trade.edge,
        market_role=paper_trade.market_role,
        strategy_type=paper_trade.strategy_type,
        p_flip_effective=paper_trade.p_flip_effective,
        p_win_effective=paper_trade.p_win_effective,
        stop_loss_pct=paper_trade.stop_loss_pct,
        stop_loss_price=paper_trade.stop_loss_price,
        take_profit_enabled=paper_trade.take_profit_enabled,
        take_profit_multiplier=paper_trade.take_profit_multiplier,
        take_profit_price=paper_trade.take_profit_price,
        position_status="OPEN",
        position_accounting_version=0,    # инициализируется после FILLED
        model_key=paper_trade.model_key,
        confirm_model_key=paper_trade.confirm_model_key,
        confirm_model_version=paper_trade.confirm_model_version,
        model_attribution_source=paper_trade.model_attribution_source,
        config_snapshot=paper_trade.config_snapshot,
        market_end_time=paper_trade.market_end_time,
        source_paper_trade_id=paper_trade.id,  # ← ключевая ссылка на источник
        created_at=now,
        updated_at=now,
    )


def _build_live_request(
    candidate: LiveMirrorCandidate,
    paper_request: ExecutionRequest,
    live_trade: TradeHistory,
    now: datetime,
    target_mode: str,
) -> ExecutionRequest:
    """
    Создаёт новый ExecutionRequest(requested_mode=target_mode, state='READY').
    source_paper_request_id = paper_request.id
    """
    return ExecutionRequest(
        id=uuid.uuid4(),
        idempotency_key=f"{target_mode}-OPEN-mirror-{candidate.id}",
        requested_mode=target_mode,
        trade_history_id=live_trade.id,         # ← ID только что созданной LIVE-строки
        intent="OPEN",
        trigger_reason="MIRROR",
        market_id=paper_request.market_id,
        asset=paper_request.asset,
        outcome_to_buy=paper_request.outcome_to_buy,
        target_amount_usdc=paper_request.target_amount_usdc,
        requested_shares=paper_request.requested_shares,
        limit_price=paper_request.limit_price,
        max_slippage_pct=paper_request.max_slippage_pct,
        max_spend_usdc=paper_request.max_spend_usdc,
        ttl_seconds=paper_request.ttl_seconds,
        state="READY",
        source_paper_request_id=paper_request.id,  # ← ключевая ссылка на источник
        created_at=now,
        updated_at=now,
    )


# ── Основная логика выпуска ───────────────────────────────────────────────────

async def release_batch(
    session: AsyncSession,
    target_mode: str,
) -> int:
    """
    Выпускает очередную порцию кандидатов.
    Возвращает количество успешно выпущенных.

    Каждый кандидат обрабатывается в отдельной транзакции —
    ошибка на одном не блокирует остальные.

    ЗАПРЕЩЕНО:
    - Изменять PAPER TradeHistory или PAPER ExecutionRequest
    - Создавать ExecutionRequest(state!='READY')
    """
    release_mode = await _get_release_mode(session)
    if release_mode == "DISABLED":
        return 0

    candidates = await _get_eligible_candidates(session, release_mode, target_mode)
    if not candidates:
        return 0

    released = 0

    for candidate in candidates:
        try:
            await _release_one(session, candidate, target_mode)
            released += 1
        except Exception as exc:
            # Откат только этой строки; остальные кандидаты обрабатываются независимо
            await session.rollback()
            logger.error(
                "release_failed for candidate %s: %s",
                candidate.id,
                exc,
                exc_info=True,
            )

    if released:
        logger.info(
            "release_gate: выпущено %d кандидатов (mode=%s, target=%s)",
            released, release_mode, target_mode,
        )
    return released


async def _release_one(
    session: AsyncSession,
    candidate: LiveMirrorCandidate,
    target_mode: str,
) -> None:
    """
    Атомарно:
    1. Загружает исходные PAPER-строки (read-only)
    2. Создаёт TradeHistory(mode=target_mode)
    3. Создаёт ExecutionRequest(mode=target_mode, state='READY')
    4. Обновляет LiveMirrorCandidate.state = 'RELEASED'
    5. Коммитит

    Если шаг 3 или 4 падает — откат, PAPER-строки остаются неизменными.
    """
    now = datetime.now(timezone.utc)

    # Загружаем PAPER-источники (только чтение)
    paper_request = (await session.execute(
        select(ExecutionRequest).where(
            ExecutionRequest.id == candidate.source_paper_request_id
        )
    )).scalar_one_or_none()

    if paper_request is None:
        candidate.state = "REJECTED"
        candidate.rejection_reason = "source_paper_request not found"
        await session.commit()
        logger.warning("Rejected candidate %s: source_paper_request not found", candidate.id)
        return

    paper_trade = (await session.execute(
        select(TradeHistory).where(
            TradeHistory.id == candidate.source_paper_trade_id
        )
    )).scalar_one_or_none()

    if paper_trade is None:
        candidate.state = "REJECTED"
        candidate.rejection_reason = "source_paper_trade not found"
        await session.commit()
        logger.warning("Rejected candidate %s: source_paper_trade not found", candidate.id)
        return

    # Создаём LIVE-строки
    live_trade = _build_live_trade(candidate, paper_trade, now, target_mode)
    session.add(live_trade)
    await session.flush()  # получаем live_trade.id до создания request

    live_request = _build_live_request(candidate, paper_request, live_trade, now, target_mode)
    session.add(live_request)
    await session.flush()

    # Обновляем кандидата (атомарно в той же транзакции)
    candidate.state = "RELEASED"
    candidate.released_at = now
    candidate.released_trade_id = live_trade.id
    candidate.released_request_id = live_request.id

    await session.commit()

    logger.info(
        "release_gate: candidate=%s → live_trade=%d live_request=%s",
        candidate.id,
        live_trade.id,
        live_request.id,
    )


# ── Главный цикл ──────────────────────────────────────────────────────────────

async def run_gate(target_mode: str) -> None:
    """Главный цикл release_gate."""
    global _shutdown

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    logger.info(
        "release_gate запущен. target_mode=%s POLL_INTERVAL=%ss",
        target_mode, POLL_INTERVAL,
    )

    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    while not _shutdown:
        try:
            async with Session() as session:
                await release_batch(session, target_mode)
        except Exception:
            logger.exception("Ошибка в release_batch, продолжаем через %ss", POLL_INTERVAL)
        await asyncio.sleep(POLL_INTERVAL)

    logger.info("release_gate остановлен.")
    await engine.dispose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Release gate: LiveMirrorCandidate → LIVE ExecutionRequest")
    parser.add_argument(
        "--mode",
        choices=["SHADOW", "LIVE"],
        default="SHADOW",
        help="Целевой режим для LIVE-заявок (SHADOW или LIVE)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        asyncio.run(run_gate(args.mode))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
