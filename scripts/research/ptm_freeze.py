"""PTM item 3: freeze DB inputs into immutable files with hashes.

Dumps market_snapshots rows for the universe + expirations used, in chunks.
Output: artifacts/research/price_time_map/_freeze/<name>/{meta.json,chunks/*.csv.gz}
Usage: python3 ptm_freeze.py <name>   (reads market list from argv file or all expirations)
"""
import asyncio
import csv
import gzip
import hashlib
import json
import os
import sys

QUERY = (
    "SELECT market_id, asset, recorded_at, time_left_min, mid_price, best_bid, best_ask, spread,"
    " final_outcome FROM market_snapshots WHERE market_id = ANY($1::text[])"
    " ORDER BY market_id, recorded_at"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def main():
    import asyncpg
    name = sys.argv[1]
    if sys.argv[2] == "EXP":
        exp_all = json.load(open("artifacts/research/market_expirations.json", encoding="utf-8"))
        mids = sorted(exp_all.keys())
    elif sys.argv[2] == "FUNNEL":
        obs_doc = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))
        mids = sorted({str(o["market_id"]) for o in obs_doc["observations"] if o.get("asset") in ("BTC", "ETH", "SOL", "XRP", "DOGE")})
    else:
        mids = [line.strip() for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
    outdir = os.path.join("artifacts", "research", "price_time_map", "_freeze", name)
    os.makedirs(os.path.join(outdir, "chunks"), exist_ok=True)
    con = await asyncpg.connect(user="polyflip", password=os.environ.get("PTM_PGPASSWORD", ""), database="polyflip", host="localhost")
    files = []
    total = 0
    try:
        max_id = await con.fetchval("SELECT max(id) FROM market_snapshots")
        total_rows = await con.fetchval("SELECT count(*) FROM market_snapshots")
        B = 1000
        for i in range(0, len(mids), B):
            chunk = mids[i:i + B]
            rows = await con.fetch(QUERY, chunk)
            path = os.path.join(outdir, "chunks", "part_%04d.csv.gz" % (i // B))
            with gzip.open(path, "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["market_id", "asset", "recorded_at", "time_left_min", "mid_price", "best_bid", "best_ask", "spread", "final_outcome"])
                for r in rows:
                    w.writerow([r["market_id"], r["asset"], r["recorded_at"].isoformat(), r["time_left_min"], r["mid_price"], r["best_bid"], r["best_ask"], r["spread"], r["final_outcome"]])
            files.append({"path": os.path.relpath(path, outdir), "sha256": sha256_file(path), "rows": len(rows)})
            total += len(rows)
            print("freeze chunk %d/%d rows=%d" % (i // B + 1, (len(mids) + B - 1) // B, total), flush=True)
    finally:
        await con.close()
    meta = {"name": name, "markets_requested": len(mids), "rows": total,
            "db_snapshot_rows_info": total_rows, "db_snapshot_max_id_info": max_id,
            "query": QUERY, "files": files}
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=2)
    print("FREEZE_DIR=" + outdir, "rows=", total)


asyncio.run(main())
