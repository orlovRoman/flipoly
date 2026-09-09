"""Probe snapshot-side quote + underlying coverage for a sample of markets (read-only)."""
import asyncio
import json
import os

SAMPLE = 400


async def main():
    import asyncpg
    obs = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))["observations"]
    mids = sorted({str(o["market_id"]) for o in obs})[:SAMPLE]
    con = await asyncpg.connect(user="polyflip", password=os.environ.get("PTM_PGPASSWORD", ""), database="polyflip", host="localhost")
    try:
        rows = await con.fetch(
            """SELECT market_id, recorded_at, time_left_min, mid_price,
                      poly_up_best_bid, poly_up_best_ask, poly_down_best_bid, poly_down_best_ask,
                      best_bid, best_ask, binance_spot_mid, final_outcome
               FROM market_snapshots WHERE market_id = ANY($1::text[])""",
            mids,
        )
        print("snapshot rows:", len(rows))
        import collections
        def cov(col):
            return sum(1 for r in rows if r[col] is not None)
        for c in ("poly_up_best_bid", "poly_up_best_ask", "poly_down_best_bid", "poly_down_best_ask", "best_bid", "best_ask", "binance_spot_mid", "mid_price", "time_left_min", "final_outcome"):
            print(f"{c}: {cov(c)}/{len(rows)}")
        per = collections.Counter(r["market_id"] for r in rows)
        import statistics
        print("per-market rows: min", min(per.values()), "median", statistics.median(per.values()), "max", max(per.values()))
    finally:
        await con.close()


asyncio.run(main())
