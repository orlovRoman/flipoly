"""
scripts/research/fetch_crypto_candles_1m.py

Fetches 1-minute historical candles from Binance public CDN API and saves
to artifacts/research/crypto_candles_1m.csv for Stage 2 research.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polyflip.crypto.binance_client import fetch_klines_range

DEFAULT_SYMBOLS = ["BTCUSDT"]
START_DT = datetime(2026, 6, 25, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(2026, 9, 9, 1, 0, tzinfo=timezone.utc)


def fetch_symbol_candles_1m(
    symbol: str,
    start_dt: datetime = START_DT,
    end_dt: datetime = END_DT,
    sleep_sec: float = 0.05,
) -> pd.DataFrame:
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)
    
    print(f"Fetching 1m candles for {symbol} from {start_dt.isoformat()} to {end_dt.isoformat()}...")
    t0 = time.time()
    
    candles_iter = fetch_klines_range(
        symbol=symbol,
        interval="1m",
        since_ms=since_ms,
        until_ms=until_ms,
        sleep_sec=sleep_sec,
    )
    
    rows = []
    for c in candles_iter:
        rows.append({
            "symbol": symbol,
            "interval": "1m",
            "open_time": c["open_time"].strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
        })
    
    elapsed = time.time() - t0
    print(f"Fetched {len(rows)} candles for {symbol} in {elapsed:.1f}s")
    df = pd.DataFrame(rows)
    return df


def main():
    out_path = REPO_ROOT / "artifacts" / "research" / "crypto_candles_1m.csv"
    dfs = []
    for sym in DEFAULT_SYMBOLS:
        df = fetch_symbol_candles_1m(sym)
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(full_df)} rows to {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
