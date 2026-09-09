"""
Stage 1 (Step 2 of protocol): immutable data export from the production DB.

Runs ON the server inside the worktree. Writes frozen, gzipped CSVs under
artifacts/research/strike_baseline/data_export/<timestamp>/raw/*.csv.gz
plus export_manifest.json with SHA-256 of every file, export time, and row counts.

The export is the single immutable input for all compared models in this run.
Raw files are gzipped copies of a \\copy snapshot; the manifest is written last
and contains the hash of every artifact including gz files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_CONTAINER = "polyflip_db"
START_SQL = "2026-06-25T00:00:00"
END_SQL = "2026-09-09T23:59:59"
PERIOD = {"start_utc": START_SQL, "end_utc": END_SQL}

# Provide explicit bound pair eliminating interval arithmetic in SQL.
CANDLE_START_LO = "2026-06-23T00:00:00"  # START_SQL - 2 days padding
CANDLE_END_HI = "2026-09-10T01:59:59"    # END_SQL + 2 hours padding

# (name, sql) — copied to /tmp/<name>.csv inside container then gzipped out.
EXPORT_TABLES = {
    "markets": """
        SELECT market_id, asset, question, slug, condition_id, status,
               end_time_est, market_start_at, market_end_at,
               current_yes_price, current_no_price, current_spread,
               underlying_price, strike_value, strike_source,
               strike_effective_at, strike_received_at, strike_observed_at,
               oracle_price, binance_price, settlement_price_source,
               resolution_status, final_outcome, resolved_at,
               resolution_source, resolution_checked_at,
               trading_status, accepting_orders, volume_status,
               yes_token_id, no_token_id, created_at, updated_at
        FROM live_markets
        ORDER BY market_id;
    """,
    "candles_15m": """
        SELECT symbol, "interval", open_time, close_time, is_closed,
               open, high, low, close, volume, taker_buy_volume, source
        FROM crypto_candles
        WHERE "interval" = '15m'
          AND open_time >= %(clo)s
          AND open_time <= %(chi)s
        ORDER BY symbol, open_time;
    """,
    "underlying_observations": """
        SELECT instrument, source, event_at, price, received_at, extra_data
        FROM underlying_observations
        ORDER BY instrument, event_at;
    """,
}

# Snapshots are 2.9 GB — export in time chunks (inclusive, read-only) to avoid OOM.
SNAPSHOT_CHUNK_DAYS = 5
SNAPSHOT_QUERY = """
SELECT market_id, asset, recorded_at, best_bid, best_ask
FROM market_snapshots
WHERE recorded_at >= %(cstart)s AND recorded_at < %(cend)s
ORDER BY market_id, recorded_at;
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_copy(db_container: str, sql: str, out_csv_gz: Path, params: dict | None = None):
    """Stream a \\copy ... TO STDOUT from inside the container to a gzipped host file.

    Avoids the container filesystem and OOM pressure: psql prints the CSV to
    stdout, docker exec -i passes quietly to the host. The query string is fed
    as psql input via /dev/stdin.
    """
    sql_full = " ".join(sql.split()).rstrip(";")
    if params:
        for k, v in params.items():
            sql_full = sql_full.replace("%(" + k + ")s", "'" + v + "'")
    psql_sql = ("\\copy (" + sql_full + ") TO STDOUT WITH (FORMAT CSV, HEADER true)\n").encode("utf-8")
    proc = subprocess.run(
        [
            "docker", "exec", "-i", db_container,
            "sh", "-c",
            "psql -U polyflip -d polyflip -f /dev/stdin",
        ],
        input=psql_sql, check=True, capture_output=True,
    )
    with gzip.open(out_csv_gz, "wb", compresslevel=6) as fout:
        fout.write(proc.stdout)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = root / "artifacts" / "research" / "strike_baseline" / "data_export" / ts
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "period": PERIOD,
        "db_container": DB_CONTAINER,
        "files": {},
        "row_counts": {},
    }

    params = {"start": START_SQL, "end": END_SQL, "clo": CANDLE_START_LO, "chi": CANDLE_END_HI}
    for name, sql in EXPORT_TABLES.items():
        out_gz = raw_dir / f"{name}.csv.gz"
        print(f"[export] {name} ...", flush=True)
        run_copy(DB_CONTAINER, sql, out_gz, params)
        manifest["files"][f"raw/{name}.csv.gz"] = {"sha256": sha256_file(out_gz), "bytes": out_gz.stat().st_size}
        print(f"  -> {out_gz.stat().st_size/1e6:.1f} MB", flush=True)

    # Snapshots: chunked by day-window to stay within container memory.
    print("[export] snapshots (chunked) ...", flush=True)
    snap_gz = raw_dir / "snapshots.csv.gz"
    cstart = datetime.fromisoformat(START_SQL)
    cend = datetime.fromisoformat(END_SQL)
    total_rows = 0
    with gzip.open(snap_gz, "wb", compresslevel=6) as fout:
        wrote_header = False
        cur = cstart
        while cur < cend:
            nxt = min(cur + timedelta(days=SNAPSHOT_CHUNK_DAYS), cend)
            chunk_gz = Path(tempfile.mkdtemp(prefix="sb_snap_")) / "chunk.csv.gz"
            run_copy(
                DB_CONTAINER, SNAPSHOT_QUERY,
                chunk_gz,
                {"cstart": cur.isoformat(timespec="seconds"), "cend": nxt.isoformat(timespec="seconds")},
            )
            with gzip.open(chunk_gz, "rt", encoding="utf-8") as fin:
                header = fin.readline()
                if not wrote_header:
                    fout.write(header.encode("utf-8"))
                    wrote_header = True
                for line in fin:
                    fout.write(line.encode("utf-8"))
                    total_rows += 1
            chunk_gz.unlink(missing_ok=True)
            print(f"  chunk {cur.date()}..{nxt.date()} done (rows so far {total_rows})", flush=True)
            cur = nxt
    manifest["files"][f"raw/snapshots.csv.gz"] = {
        "sha256": sha256_file(snap_gz),
        "bytes": snap_gz.stat().st_size,
    }
    manifest["row_counts"]["snapshots"] = total_rows
    print(f"  -> snapshots.csv.gz {snap_gz.stat().st_size/1e6:.1f} MB ({total_rows} rows)", flush=True)

    # row counts (uncompressed, via gzopen)
    for name in EXPORT_TABLES:
        p = raw_dir / f"{name}.csv.gz"
        with gzip.open(p, "rt", encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        manifest["row_counts"][name] = n
        print(f"  rows[{name}] = {n}", flush=True)

    manifest_path = out_dir / "export_manifest.json"
    canonical = json.dumps(manifest, indent=2, sort_keys=True).replace("\r\n", "\n")
    manifest["manifest_file_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    final = json.dumps(manifest, indent=2, sort_keys=True).replace("\r\n", "\n")
    manifest_path.write_bytes((final + "\n").encode("utf-8"))

    print()
    print(f"EXPORT_COMPLETE dir={out_dir}")
    print(f"  manifest_sha256={manifest['manifest_file_sha256']}")


if __name__ == "__main__":
    main()