"""
Клиент Binance для получения OHLCV-свечей.

Endpoint: GET https://data-api.binance.vision/api/v3/klines
- Не требует API-ключа.
- Работает как readonly CDN — не зависит от торгового контура Binance.
- Возвращает до 1000 свечей за запрос.
- Rate limit: 6000 weight/min, klines = 2 weight → ~3000 req/min (практически неограниченно).

Маппинг символов: 'bitcoin' → 'BTCUSDT', 'ethereum' → 'ETHUSDT'
"""
from __future__ import annotations

import time
import httpx
from httpx import HTTPTransport
from datetime import datetime, timezone
from typing import Iterator

BINANCE_BASE = "https://data-api.binance.vision"
KLINES_LIMIT = 1000   # максимум за один запрос

# human-readable name → Binance symbol
COIN_TO_SYMBOL: dict[str, str] = {
    "bitcoin":  "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "ripple":   "XRPUSDT",
    "solana":   "SOLUSDT",
}

_transport = HTTPTransport(retries=3)   # автоматический retry на network errors

def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = KLINES_LIMIT,
) -> list[dict]:
    """
    Запрашивает свечи из Binance.

    symbol:   'BTCUSDT' (Binance формат, НЕ 'bitcoin')
    interval: '1m' | '5m' | '15m' | '1h' | '4h'
    start_ms: начало диапазона в UTC milliseconds (включительно)
    end_ms:   конец диапазона в UTC milliseconds (включительно)

    Возвращает список словарей с ключами:
      open_time, open, high, low, close, volume, taker_buy_volume
    """
    params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    with httpx.Client(
        transport=_transport,
        timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)
    ) as client:
        resp = client.get(f"{BINANCE_BASE}/api/v3/klines", params=params)
        resp.raise_for_status()

    raw: list[list] = resp.json()

    candles = []
    for row in raw:
        candles.append({
            "open_time":        datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            "open":             float(row[1]),
            "high":             float(row[2]),
            "low":             float(row[3]),
            "close":            float(row[4]),
            "volume":           float(row[5]),
            "taker_buy_volume": float(row[9]),
        })
    return candles


def fetch_klines_range(
    symbol: str,
    interval: str,
    since_ms: int,
    until_ms: int | None = None,
    sleep_sec: float = 0.1,
) -> Iterator[dict]:
    """
    Итерирует все свечи в диапазоне [since_ms, until_ms] батчами по 1000.
    Используется для backfill и догрузки пропущенных свечей.
    """
    until_ms = until_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    current_ms = since_ms

    while current_ms < until_ms:
        batch = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_ms=current_ms,
            end_ms=until_ms,
            limit=KLINES_LIMIT,
        )
        if not batch:
            break

        for candle in batch:
            yield candle

        # следующий батч начинается с open_time последней свечи + 1ms
        last_ts = int(batch[-1]["open_time"].timestamp() * 1000)
        current_ms = last_ts + 1

        if len(batch) < KLINES_LIMIT:
            break  # дошли до конца

        time.sleep(sleep_sec)
