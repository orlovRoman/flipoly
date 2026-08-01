"""
Bulk-загрузка исторических свечей из Binance.

Использует уже существующий fetch_klines_range() из binance_client.py.
Идемпотентен: повторный вызов за тот же период вернёт 0 (ON CONFLICT DO NOTHING).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.binance_client import fetch_klines_range
from polyflip.crypto.candle_repository import upsert_candles

logger = structlog.get_logger(__name__)

# Настройки по умолчанию — переопределяются через аргументы или RuntimeSettings
DEFAULT_SYMBOLS:   list[str] = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XRPUSDT", "SOLUSDT"]
DEFAULT_INTERVALS: list[str] = ["5m", "15m"]
DEFAULT_DAYS_BACK: int       = 90
BINANCE_SLEEP_SEC: float     = 0.15   # 150ms между батчами → ~6 req/sec, хорошо в рамках лимита


async def load_history(
    db:          AsyncSession,
    symbol:      str,
    interval:    str,
    days_back:   int   = DEFAULT_DAYS_BACK,
    sleep_sec:   float = BINANCE_SLEEP_SEC,
) -> int:
    """
    Скачивает свечи за последние days_back дней и записывает в БД.

    Возвращает количество вставленных строк (0 при повторном вызове).
    Синхронный fetch_klines_range запускается в отдельном thread,
    чтобы не блокировать event loop.
    """
    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    logger.info(
        "historical_load_start",
        symbol=symbol, interval=interval,
        days_back=days_back, since=datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat(),
    )

    # fetch_klines_range — синхронный итератор → запускаем через to_thread
    def _fetch_all() -> list[dict]:
        return list(fetch_klines_range(
            symbol=symbol,
            interval=interval,
            since_ms=since_ms,
            until_ms=now_ms,
            sleep_sec=sleep_sec,
        ))

    candles = await asyncio.to_thread(_fetch_all)
    inserted = await upsert_candles(db, symbol, interval, candles)

    logger.info(
        "historical_load_done",
        symbol=symbol, interval=interval,
        fetched=len(candles), inserted=inserted,
    )
    return inserted


async def load_history_all(
    db:        AsyncSession,
    symbols:   list[str] = DEFAULT_SYMBOLS,
    intervals: list[str] = DEFAULT_INTERVALS,
    days_back: int       = DEFAULT_DAYS_BACK,
) -> dict[str, int]:
    """
    Загружает историю для всех символов и интервалов.
    Запускает последовательно (не параллельно) — один клиент к Binance CDN.

    Возвращает: {"BTCUSDT_5m": 25920, "BTCUSDT_15m": 8640, ...}
    """
    results: dict[str, int] = {}
    for symbol in symbols:
        for interval in intervals:
            key = f"{symbol}_{interval}"
            try:
                results[key] = await load_history(db, symbol, interval, days_back)
            except Exception:
                logger.exception("historical_load_error", symbol=symbol, interval=interval)
                results[key] = -1   # -1 = ошибка, не путать с 0 (уже загружено)
    return results

if __name__ == "__main__":
    import argparse
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import os

    parser = argparse.ArgumentParser(description="Загрузка исторических свечей из Binance.")
    parser.add_argument("--days", type=int, default=30, help="Количество дней истории (default: 30)")
    args = parser.parse_args()

    async def main():
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("ERROR: DATABASE_URL environment variable is not set.")
            import sys
            sys.exit(1)
            
        engine = create_async_engine(db_url)
        SessionLocal = async_sessionmaker(engine)
        try:
            async with SessionLocal() as session:
                logger.info("historical_load_cli_start", days=args.days, db=db_url)
                res = await load_history_all(session, days_back=args.days)
                logger.info("historical_load_cli_done", results=res)
                await session.commit()
        finally:
            await engine.dispose()
    
    asyncio.run(main())
