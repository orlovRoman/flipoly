"""
Удаление устаревших свечей из БД.
Запускается раз в сутки из scheduler/jobs.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import CryptoCandle

logger = structlog.get_logger(__name__)

DEFAULT_RETENTION_DAYS = 90


async def prune_old_candles(
    db:             AsyncSession,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> int:
    """
    Удаляет свечи с open_time < (now - retention_days).

    Безопасно запускать параллельно с инкрементальным сборщиком —
    удаляет только строки старше retention_days, сборщик пишет свежие.

    Возвращает количество удалённых строк.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    result = await db.execute(
        delete(CryptoCandle).where(CryptoCandle.open_time < cutoff)
    )
    await db.commit()

    deleted = result.rowcount
    logger.info("candle_pruner_done", deleted=deleted, cutoff=cutoff.isoformat(), retention_days=retention_days)
    return deleted
