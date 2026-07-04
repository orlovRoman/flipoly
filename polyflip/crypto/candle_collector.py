"""
Инкрементальный сборщик свечей.
Запрашивает только новые свечи — с момента последней записи в БД.
Вызывается из scheduler/jobs.py каждые 15 минут.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.binance_client import fetch_klines, COIN_TO_SYMBOL
from polyflip.crypto.candle_repository import upsert_candles, get_latest_open_time

log = logging.getLogger(__name__)

SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
INTERVAL  = "15m"


async def collect_new_candles(session: AsyncSession) -> dict[str, int]:
    """
    Для каждого символа:
      1. Читает open_time последней свечи из БД.
      2. Запрашивает у Binance свечи начиная с этого времени.
      3. Вставляет только новые (upsert ON CONFLICT DO NOTHING).

    Принимает готовую session — как все jobs в scheduler/jobs.py.
    Возвращает {symbol: inserted_count}.
    """
    results: dict[str, int] = {}

    for symbol in SYMBOLS:
        latest = await get_latest_open_time(session, symbol, INTERVAL)

        if latest:
            # Берём с момента последней свечи — Binance вернёт её саму и всё новее
            start_ms = int(latest.timestamp() * 1000)
        else:
            # Первый запуск без backfill — берём последние 24 часа
            start_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000
            )

        candles = fetch_klines(symbol, INTERVAL, start_ms=start_ms)

        # Фильтруем: исключаем свечу с open_time == latest (она уже есть)
        if latest:
            candles = [c for c in candles if c["open_time"] > latest]

        inserted = await upsert_candles(session, symbol, INTERVAL, candles)
        log.info("candle_collector_done", symbol=symbol, inserted=inserted)
        results[symbol] = inserted

    return results
