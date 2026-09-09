"""Probe MarketSnapshot coverage for research markets (read-only)."""
import asyncio
import json
import os


async def main():
    try:
        import asyncpg
    except ImportError:
        print("NO_ASYNCPG")
        return
    obs = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))["observations"]
    mids = sorted({str(o["market_id"]) for o in obs})
    print("markets:", len(mids))
    try:
        con = await asyncio.wait_for(
            asyncpg.connect(user="polyflip", password=os.environ.get("PTM_PGPASSWORD", ""), database="polyflip", host="localhost"), 10
        )
    except Exception as exc:
        print("CONNECT_FAIL", type(exc).__name__, str(exc)[:120])
        return
    try:
        n = await con.fetchval("SELECT count(*) FROM market_snapshots")
        print("snapshots total:", n)
        cols = await con.fetch("SELECT column_name FROM information_schema.columns WHERE table_name='market_snapshots'")
        print("cols:", sorted(r["column_name"] for r in cols))
        hit = await con.fetchval(
            "SELECT count(DISTINCT market_id) FROM market_snapshots WHERE market_id = ANY($1::text[])",
            mids[:2000],
        )
        print("distinct research markets with snapshots (first 2000):", hit)
        per = await con.fetch(
            "SELECT market_id, count(*) c FROM market_snapshots WHERE market_id = ANY($1::text[]) GROUP BY 1 ORDER BY c DESC LIMIT 5",
            mids[:2000],
        )
        print("top counts:", [(r["market_id"], r["c"]) for r in per])
    finally:
        await con.close()


asyncio.run(main())
