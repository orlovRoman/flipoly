"""
Загружает историю свечей для BTC и ETH из Binance.
Запускать ОДИН раз вручную перед первым деплоем.

Usage:
    python scripts/backfill_candles.py
    python scripts/backfill_candles.py --days 30 --symbols BTCUSDT
    python scripts/backfill_candles.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from polyflip.crypto.binance_client import fetch_klines_range
from polyflip.crypto.candle_repository import upsert_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["15m"]


async def backfill(symbols: list[str], intervals: list[str], days: int, dry_run: bool) -> None:
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    engine = create_async_engine(db_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    for symbol in symbols:
        for interval in intervals:
            log.info(f"📥 Backfill {symbol} {interval} | last {days} days")
            candles = list(fetch_klines_range(symbol, interval, since_ms=since_ms))
            log.info(f"   Получено: {len(candles)} свечей")

            if dry_run:
                log.info(f"   [DRY-RUN] Вставка пропущена")
                if candles:
                    log.info(f"   Первая: {candles[0]['open_time']}  Последняя: {candles[-1]['open_time']}")
                continue

            async with Session() as session:
                inserted = await upsert_candles(session, symbol, interval, candles)
            log.info(f"   ✅ Вставлено: {inserted} | дубликатов: {len(candles) - inserted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--intervals", nargs="+", default=INTERVALS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(backfill(args.symbols, args.intervals, args.days, args.dry_run))
