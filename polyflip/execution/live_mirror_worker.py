"""
live_mirror_worker.py — Этап 5: зеркалирование исполненных PAPER OPEN → LiveMirrorCandidate

Инварианты (запрещено в этом модуле):
  - Изменять TradeHistory.status, .position_status, .pnl, .realized_pnl_usdc
  - Изменять ExecutionRequest.state
  - Создавать TradeHistory(mode='LIVE') или ExecutionRequest(mode='LIVE')
  - Читать или изменять что-либо режима LIVE

Поведение:
  - Запускается бесконечным циклом, шаг каждые POLL_INTERVAL секунд.
  - Берёт только PAPER OPEN заявки в финальных состояниях FILLED / PARTIALLY_FILLED_FINAL.
  - Для каждой создаёт LiveMirrorCandidate(state='NEW', target_mode=SHADOW) через
    INSERT ... ON CONFLICT DO NOTHING  →  идемпотентно при перезапуске.
  - Не трогает PAPER-строки.

Управление:
  LIVE_MIRROR_ENABLED=true   — воркер активен
  LIVE_MIRROR_ENABLED=false  — воркер спит, не создаёт кандидатов
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select, exists
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from polyflip.db.execution_models import ExecutionRequest, LiveMirrorCandidate
from polyflip.db.models import TradeHistory, RuntimeSettings

logger = logging.getLogger("live_mirror_worker")

# ── Настройки ────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")  # lazy-checked in run_worker()
POLL_INTERVAL: float = float(os.environ.get("MIRROR_POLL_INTERVAL", "10"))
BATCH_SIZE: int = int(os.environ.get("MIRROR_BATCH_SIZE", "50"))
TARGET_MODE: str = os.environ.get("MIRROR_TARGET_MODE", "SHADOW")  # SHADOW до Этапа 10

# Состояния PAPER OPEN-заявок, которые считаются финально исполненными
PAPER_MIRRORABLE_STATES: frozenset[str] = frozenset({
    "FILLED",
    "PARTIALLY_FILLED_FINAL",
})

# ── Главный цикл ──────────────────────────────────────────────────────────────

_shutdown = False


def _handle_sigterm(signum: int, frame: object) -> None:
    global _shutdown
    logger.info("Получен сигнал %s, выполняем плановую остановку...", signum)
    _shutdown = True


async def runtime_bool(
    session: AsyncSession,
    key: str,
    default: bool = False,
) -> bool:
    """Читает флаг из RuntimeSettings в БД в реальном времени."""
    value = await session.scalar(
        select(RuntimeSettings.value).where(RuntimeSettings.key == key)
    )
    if value is None:
        return default
    return value.strip().lower() == "true"


async def mirror_batch(session: AsyncSession) -> int:
    """
    Зеркалирует одну порцию PAPER OPEN → LiveMirrorCandidate.
    Возвращает количество новых кандидатов.

    Запрещено внутри:
    - Изменять paper_request.state
    - Изменять paper_trade.*
    - Создавать LIVE TradeHistory или LIVE ExecutionRequest
    """
    # Курсор начала зеркалирования
    started_at_raw = await session.scalar(
        select(RuntimeSettings.value).where(RuntimeSettings.key == "LIVE_MIRROR_STARTED_AT")
    )
    started_at = None
    if started_at_raw:
        try:
            started_at = datetime.fromisoformat(started_at_raw)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # Выбираем исполненные PAPER OPEN без существующего кандидата
    conditions = [
        ExecutionRequest.requested_mode == "PAPER",
        ExecutionRequest.intent == "OPEN",
        ExecutionRequest.state.in_(PAPER_MIRRORABLE_STATES),
        TradeHistory.mode == "PAPER",
        # Идемпотентность: пропускаем уже зеркалированные
        ~exists().where(
            LiveMirrorCandidate.source_paper_request_id == ExecutionRequest.id,
            LiveMirrorCandidate.target_mode == TARGET_MODE,
        ),
    ]

    if started_at:
        conditions.append(ExecutionRequest.updated_at >= started_at)

    stmt = (
        select(ExecutionRequest, TradeHistory)
        .join(
            TradeHistory,
            TradeHistory.id == ExecutionRequest.trade_history_id,
        )
        .where(*conditions)
        .order_by(ExecutionRequest.updated_at)
        .limit(BATCH_SIZE)
    )

    rows = (await session.execute(stmt)).all()

    if not rows:
        return 0

    created = 0
    now = datetime.now(timezone.utc)

    for paper_request, paper_trade in rows:
        snapshot = _build_signal_snapshot(paper_request, paper_trade)
        signal_hash = _compute_hash(snapshot)

        insert_stmt = (
            pg_insert(LiveMirrorCandidate)
            .values(
                source_paper_request_id=paper_request.id,
                source_paper_trade_id=paper_trade.id,
                target_mode=TARGET_MODE,
                state="NEW",
                signal_snapshot=snapshot,
                signal_hash=signal_hash,
                created_at=now,
            )
            .on_conflict_do_nothing(
                index_elements=["source_paper_request_id", "target_mode"],
            )
        )

        result = await session.execute(insert_stmt)
        if result.rowcount:
            created += 1
            logger.debug(
                "Создан кандидат: paper_request=%s paper_trade=%d signal_hash=%s",
                paper_request.id,
                paper_trade.id,
                signal_hash[:12],
            )

    if created:
        await session.commit()
        logger.info("Зеркалировано %d PAPER OPEN → LiveMirrorCandidate (target_mode=%s)", created, TARGET_MODE)
    else:
        # Все строки уже были зеркалированы (race condition / повтор)
        await session.rollback()

    return created


def _build_signal_snapshot(
    paper_request: ExecutionRequest,
    paper_trade: TradeHistory,
) -> dict:
    """
    Формирует детерминированный снимок сигнала из PAPER-данных.
    Содержит только данные сигнала — никаких состояний, PnL, timestamps.
    """
    return {
        "market_id": paper_request.market_id,
        "asset": paper_request.asset,
        "outcome_to_buy": paper_request.outcome_to_buy,
        "target_amount_usdc": str(paper_request.target_amount_usdc),
        "requested_shares": str(paper_request.requested_shares) if paper_request.requested_shares is not None else None,
        "decision_limit_price": str(paper_request.limit_price) if paper_request.limit_price is not None else None,
        "model_key": paper_trade.model_key,
        "model_version": paper_trade.model_version,
        "confirm_model_key": paper_trade.confirm_model_key,
        "confirm_model_version": paper_trade.confirm_model_version,
        "edge": paper_trade.edge,
        "market_role": paper_trade.market_role,
        "predicted_flip_prob": paper_trade.predicted_flip_prob,
        "config_snapshot": paper_trade.config_snapshot,
        "source_created_at": paper_request.created_at.isoformat() if paper_request.created_at else None,
    }


def _compute_hash(snapshot: dict) -> str:
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


async def run_worker() -> None:
    """Главный цикл mirror-воркера."""
    global _shutdown

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    logger.info(
        "live_mirror_worker запущен. TARGET_MODE=%s POLL_INTERVAL=%ss",
        TARGET_MODE,
        POLL_INTERVAL,
    )

    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    while not _shutdown:
        try:
            async with Session() as session:
                if not await runtime_bool(session, "LIVE_MIRROR_ENABLED"):
                    logger.debug("LIVE_MIRROR_ENABLED=false (в DB) — спим %ss", POLL_INTERVAL)
                else:
                    created = await mirror_batch(session)
                    if created:
                        logger.info("Батч завершён: создано %d кандидатов", created)
        except Exception:
            logger.exception("Ошибка в mirror_batch, продолжаем через %ss", POLL_INTERVAL)

        await asyncio.sleep(POLL_INTERVAL)

    logger.info("live_mirror_worker остановлен.")
    await engine.dispose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
