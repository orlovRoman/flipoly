import asyncio
from sqlalchemy import text
from polyflip.db.connection import async_session

async def run():
    async with async_session() as db:
        # Детальная диагностика OPENING с shares
        res = await db.execute(text("""
            SELECT
                id,
                market_id,
                mode,
                outcome_bought,
                entry_filled_shares,
                remaining_shares,
                entry_cost_usdc,
                realized_pnl_usdc,
                exit_reason,
                position_accounting_version,
                created_at
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'OPENING'
              AND entry_filled_shares IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """))
        print("=== Sample OPENING trades with entry_filled_shares ===")
        cols = ['id','market_id','mode','outcome','shares','remaining','cost_usdc','pnl','exit_reason','acct_ver','created_at']
        print(" | ".join(f"{c:>12}" for c in cols))
        print("-" * 140)
        for r in res.fetchall():
            vals = [str(v)[:12] for v in r]
            print(" | ".join(f"{v:>12}" for v in vals))

        # Сколько из них имеют market_id, который мы можем проверить
        res2 = await db.execute(text("""
            SELECT
                CASE
                    WHEN remaining_shares > 0 THEN 'remaining > 0'
                    WHEN remaining_shares = 0 THEN 'remaining = 0'
                    WHEN remaining_shares IS NULL THEN 'remaining NULL'
                END as remaining_cat,
                COUNT(*) as cnt
            FROM trade_history
            WHERE mode IN ('PAPER', 'SHADOW')
              AND position_status = 'OPENING'
              AND entry_filled_shares IS NOT NULL
            GROUP BY 1
        """))
        print("\n=== remaining_shares distribution for OPENING with shares ===")
        for r in res2.fetchall():
            print(f"  {r[0]:<20} cnt={r[1]}")

asyncio.run(run())
