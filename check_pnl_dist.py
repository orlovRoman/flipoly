import asyncio
from sqlalchemy import text
from polyflip.db.connection import async_session

async def run():
    async with async_session() as db:
        # Сколько CLOSED с NULL pnl
        res = await db.execute(text("""
            SELECT
                CASE
                    WHEN realized_pnl_usdc IS NULL THEN 'NULL'
                    WHEN realized_pnl_usdc = 0 THEN 'ZERO'
                    WHEN realized_pnl_usdc > 0 THEN 'POSITIVE'
                    ELSE 'NEGATIVE'
                END as pnl_category,
                COUNT(*) as cnt,
                ROUND(SUM(realized_pnl_usdc::numeric), 4) as total
            FROM trade_history
            WHERE mode = 'PAPER' AND position_status = 'CLOSED'
            GROUP BY 1
            ORDER BY cnt DESC
        """))
        print("=== PnL categories for CLOSED PAPER trades ===")
        for r in res.fetchall():
            print(f"  {r[0]:<12} cnt={r[1]:>5}  total={r[2]}")

        # Remaining OPEN with entry_filled_shares (ещё не обработанные)
        res2 = await db.execute(text("""
            SELECT COUNT(*) FROM trade_history
            WHERE position_status = 'OPEN'
              AND mode IN ('PAPER', 'SHADOW')
              AND entry_filled_shares IS NOT NULL
              AND entry_filled_shares > 0
        """))
        print(f"\nRemaining OPEN with entry_filled_shares: {res2.scalar()}")

asyncio.run(run())
