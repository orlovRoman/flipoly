import asyncio
from sqlalchemy import text
from polyflip.db.connection import async_session

async def run():
    async with async_session() as db:
        # Cat 1: CLOSED с pnl=0 — сколько имеют close_price и entry_filled_shares?
        res = await db.execute(text("""
            SELECT
                CASE
                    WHEN close_price IS NOT NULL AND entry_filled_shares IS NOT NULL THEN 'has_close_price+shares'
                    WHEN close_price IS NOT NULL AND entry_filled_shares IS NULL THEN 'has_close_price_only'
                    WHEN close_price IS NULL AND entry_filled_shares IS NOT NULL THEN 'has_shares_only'
                    ELSE 'nothing'
                END as recoverable,
                COUNT(*) as cnt
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'CLOSED'
              AND realized_pnl_usdc = 0
            GROUP BY 1
            ORDER BY cnt DESC
        """))
        print("=== CLOSED PAPER/SHADOW with pnl=0: recovery potential ===")
        for r in res.fetchall():
            print(f"  {r[0]:<35} cnt={r[1]}")

        # OPEN без entry_filled_shares — distribution
        res2 = await db.execute(text("""
            SELECT
                CASE
                    WHEN entry_filled_shares IS NULL THEN 'NULL shares'
                    WHEN entry_filled_shares = 0 THEN 'ZERO shares'
                    ELSE 'has shares'
                END as shares_status,
                COUNT(*) as cnt
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'OPEN'
            GROUP BY 1
        """))
        print("\n=== OPEN PAPER/SHADOW: entry_filled_shares ===")
        for r in res2.fetchall():
            print(f"  {r[0]:<20} cnt={r[1]}")

        # OPENING — есть ли хоть что-то?
        res3 = await db.execute(text("""
            SELECT
                CASE
                    WHEN entry_filled_shares IS NULL THEN 'NULL'
                    ELSE 'has shares'
                END as shares_status,
                COUNT(*) as cnt
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'OPENING'
            GROUP BY 1
        """))
        print("\n=== OPENING PAPER/SHADOW: entry_filled_shares ===")
        for r in res3.fetchall():
            print(f"  {r[0]:<20} cnt={r[1]}")

asyncio.run(run())
