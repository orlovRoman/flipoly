"""
Единственное место для чтения и записи CryptoCandle.
Не знает ни про Binance, ни про фичи — только БД.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from polyflip.db.models import CryptoCandle


async def upsert_candles(
    session: AsyncSession,
    symbol: str,
    interval: str,
    candles: list[dict],
) -> int:
    """
    Вставляет список свечей батчами по 1000 штук. ON CONFLICT DO NOTHING — дубликаты игнорируются.
    Возвращает количество реально вставленных строк.
    """
    if not candles:
        return 0

    rows = [
        {
            "symbol":           symbol,
            "interval":         interval,
            "open_time":        c["open_time"],
            "open":             c["open"],
            "high":             c["high"],
            "low":              c["low"],
            "close":            c["close"],
            "volume":           c.get("volume", 0.0),
            "taker_buy_volume": c.get("taker_buy_volume"),
            "source":           "binance",
        }
        for c in candles
    ]

    total_inserted = 0
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        stmt = (
            pg_insert(CryptoCandle)
            .values(batch)
            .on_conflict_do_nothing(constraint="uix_crypto_candle")
        )
        result = await session.execute(stmt)
        total_inserted += result.rowcount

    await session.commit()
    return total_inserted


async def get_recent_candles(
    session: AsyncSession,
    symbol: str,
    interval: str,
    limit: int = 200,
) -> Sequence[CryptoCandle]:
    """Последние `limit` свечей, отсортированных ASC (старые → новые)."""
    result = await session.execute(
        select(CryptoCandle)
        .where(
            CryptoCandle.symbol == symbol,
            CryptoCandle.interval == interval,
        )
        .order_by(desc(CryptoCandle.open_time))
        .limit(limit)
    )
    rows = list(result.scalars().all())
    return rows[::-1]


async def get_latest_open_time(
    session: AsyncSession,
    symbol: str,
    interval: str,
) -> datetime | None:
    """open_time самой свежей свечи или None если таблица пуста."""
    result = await session.execute(
        select(func.max(CryptoCandle.open_time))
        .where(
            CryptoCandle.symbol == symbol,
            CryptoCandle.interval == interval,
        )
    )
    return result.scalar_one_or_none()
