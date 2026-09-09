"""
Fetch Binance 1m klines for the study period (locally, heavy work on local machine).
Outputs gzipped CSVs into the same data_export/<ts>/raw/ layout with a hashed manifest.

1m candles are needed for the 60-minute rolling volatility window (vol_est_window_min).
Only CLOSED candles are trusted; open_time is the candle start (UTC).

Windowing: for decision at time D, only candles with close_time <= D are used.
Candle close_time = open_time + 1min - 1s; a candle is "closed" when now >= close_time.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
START = int(datetime(2026, 6, 5, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END = int(datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
INTERVAL = "1m"
BASE = "https://api.binance.com/api/v3/klines"
MAX_KLINES = 1000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """LF-only write so manifest hashes are platform-independent (see run_phase1)."""
    path.write_bytes(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def write_manifest(path: Path, payload: dict) -> dict:
    """manifest_file_sha256 = sha256 of canonical JSON without the self key."""
    canonical = json.dumps(payload, indent=2, sort_keys=True).replace("\r\n", "\n")
    payload = dict(payload)
    payload["manifest_file_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    final = json.dumps(payload, indent=2, sort_keys=True).replace("\r\n", "\n")
    write_text_lf(path, final + "\n")
    return payload


def fetch_symbol(symbol: str) -> list[list]:
    rows: list[list] = []
    start_ms = START
    while start_ms < END:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "limit": MAX_KLINES,
        }
        for attempt in range(5):
            try:
                r = requests.get(BASE, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        nxt = last_open + 60_000
        if nxt <= start_ms:
            break
        start_ms = nxt
        time.sleep(0.15)  # polite rate limit (≤ ~6 req/s)
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: fetch_binance_1m.py <export_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(sys.argv[1]).resolve() / "fixture_1m"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "interval": INTERVAL,
        "period_ms": {"start": START, "end": END},
        "files": {},
        "row_counts": {},
        "note": "1m BINANCE klines; only used as closed candles; open_time UTC",
    }

    for symbol in SYMBOLS:
        rows = fetch_symbol(symbol)
        rows.sort(key=lambda r: r[0])
        # kline layout: [0 open_time_ms,1 open,2 high,3 low,4 close,5 vol,6 close_time_ms,7 quote_vol,8 trades]
        csv_path = out_dir / f"{symbol}.csv.gz"
        tmp = Path(tempfile.mkdtemp(prefix="bin1m_"))
        plain = tmp / f"{symbol}.csv"
        with open(plain, "w", encoding="utf-8", newline="") as f:
            f.write("open_time_ms,open,high,low,close,volume,close_time_ms\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n")
        with open(plain, "rb") as fin, gzip.open(csv_path, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
        shutil.rmtree(tmp, ignore_errors=True)

        manifest["files"][f"fixture_1m/{symbol}.csv.gz"] = {
            "sha256": sha256_file(csv_path),
            "bytes": csv_path.stat().st_size,
        }
        manifest["row_counts"][symbol] = len(rows)
        print(f"[fetch] {symbol}: {len(rows)} candles -> {csv_path.name}", flush=True)

    manifest_path = out_dir / "binance_1m_manifest.json"
    manifest = write_manifest(manifest_path, manifest)
    print(f"BINANCE_1M_DONE manifest_sha256={manifest['manifest_file_sha256']}")


if __name__ == "__main__":
    main()