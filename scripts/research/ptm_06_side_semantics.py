"""Check best_bid/ask side semantics and binance coverage (read-only)."""
import asyncio
import json
import os


async def main():
    import asyncpg
    con = await asyncpg.connect(user="polyflip", password=os.environ.get("PTM_PGPASSWORD", ""), database="polyflip", host="localhost")
    try:
        for col in ("binance_spot_mid", "binance_perp_mid", "binance_price", "binance_spot_bid", "binance_spot_ask"):
            n = await con.fetchval(f"SELECT count(*) FROM market_snapshots WHERE {col} IS NOT NULL")
            tot = await con.fetchval("SELECT count(*) FROM market_snapshots")
            print(f"{col}: {n}/{tot}")
        rows = await con.fetch(
            """SELECT market_id, recorded_at, mid_price, best_bid, best_ask, spread, final_outcome, flip_vs_final
               FROM market_snapshots WHERE market_id='3293216' ORDER BY recorded_at LIMIT 3"""
        )
        for r in rows:
            print(dict(r))
        obs = {str(o["market_id"]): o for o in json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))["observations"]}
        o = obs.get("3293216")
        print("json row:", {k: o.get(k) for k in ("yes_ask", "no_ask", "p_market_yes", "outcome_yes", "candidate_side", "legacy_ask", "spread")})
    finally:
        await con.close()


asyncio.run(main())
