import asyncio
from sqlalchemy import text
from polyflip.db.connection import async_session

async def run():
    async with async_session() as db:
        res = await db.execute(text("""
            SELECT
                id,
                outcome_bought,
                entry_filled_shares,
                remaining_shares,
                entry_cost_usdc,
                realized_pnl_usdc
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'CLOSED'
              AND entry_filled_shares IS NOT NULL
              AND entry_filled_shares > 0
              AND realized_pnl_usdc = 0
            ORDER BY id
            LIMIT 20
        """))
        print(f"{'id':>6} {'outcome':>8} {'entry_shares':>20} {'remaining':>20} {'entry_cost':>14} {'pnl':>10}")
        print("-" * 82)
        for r in res.fetchall():
            print(f"{r[0]:>6} {r[1]:>8} {r[2]:>20} {r[3]:>20} {r[4]:>14} {r[5]:>10}")

        # Сводка по remaining_shares
        res2 = await db.execute(text("""
            SELECT
                CASE
                    WHEN remaining_shares IS NULL THEN 'NULL'
                    WHEN remaining_shares = 0 THEN 'ZERO'
                    WHEN remaining_shares > 0 THEN 'POSITIVE'
                    ELSE 'NEGATIVE'
                END as cat,
                COUNT(*) as cnt
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'CLOSED'
              AND entry_filled_shares IS NOT NULL
              AND entry_filled_shares > 0
              AND realized_pnl_usdc = 0
            GROUP BY 1
        """))
        print("\n=== remaining_shares in Cat B ===")
        for r in res2.fetchall():
            print(f"  {r[0]:<12} cnt={r[1]}")

asyncio.run(run())
