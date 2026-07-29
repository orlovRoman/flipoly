import asyncio
from sqlalchemy import select, func, text
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

async def run():
    async with async_session() as db:
        # 1. Сколько PAPER сделок теперь CLOSED с ненулевым PnL
        res = await db.execute(text("""
            SELECT
                position_status,
                COUNT(*) as cnt,
                SUM(CASE WHEN realized_pnl_usdc IS NOT NULL AND realized_pnl_usdc != 0 THEN 1 ELSE 0 END) as with_pnl,
                ROUND(SUM(realized_pnl_usdc::numeric), 4) as total_pnl
            FROM trade_history
            WHERE mode = 'PAPER'
            GROUP BY position_status
            ORDER BY cnt DESC
        """))
        rows = res.fetchall()
        print("=== PAPER trades breakdown ===")
        print(f"{'status':<20} {'count':>8} {'with_pnl':>10} {'total_pnl':>12}")
        print("-" * 55)
        for r in rows:
            print(f"{r[0]:<20} {r[1]:>8} {r[2]:>10} {r[3]:>12}")

        # 2. Примеры недавно закрытых
        res2 = await db.execute(text("""
            SELECT id, market_id, outcome_bought, realized_pnl_usdc, close_price, closed_at
            FROM trade_history
            WHERE mode = 'PAPER' AND position_status = 'CLOSED' AND realized_pnl_usdc IS NOT NULL
            ORDER BY closed_at DESC NULLS LAST
            LIMIT 10
        """))
        rows2 = res2.fetchall()
        print("\n=== Last 10 settled PAPER trades ===")
        print(f"{'id':>6} {'market_id':>10} {'outcome':>8} {'pnl_usdc':>12} {'close_price':>12} {'closed_at'}")
        print("-" * 70)
        for r in rows2:
            print(f"{r[0]:>6} {r[1]:>10} {r[2]:>8} {str(r[3]):>12} {str(r[4]):>12} {r[5]}")

asyncio.run(run())
