import asyncio
from sqlalchemy import select, func, case, text
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

async def main():
    async with async_session() as session:
        # 1. Общие показатели за 24ч для PAPER
        query_24h = text("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success_trades,
                SUM(CASE WHEN status = 'SKIPPED' THEN 1 ELSE 0 END) as skipped_trades,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_trades,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending_trades,
                SUM(COALESCE(realized_pnl_usdc, pnl, 0)) as total_pnl,
                SUM(CASE WHEN COALESCE(realized_pnl_usdc, pnl, 0) > 0 THEN 1 ELSE 0 END) as winning_trades,
                AVG(CASE WHEN status = 'SUCCESS' THEN executed_price ELSE NULL END) as avg_executed_price,
                AVG(CASE WHEN status = 'SUCCESS' THEN amount_usdc ELSE NULL END) as avg_bet_size
            FROM trade_history
            WHERE mode = 'PAPER'
              AND created_at >= NOW() - INTERVAL '24 HOURS';
        """)
        res_24h = (await session.execute(query_24h)).mappings().one()

        # 2. По активам за 24ч
        query_asset = text("""
            SELECT 
                asset,
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success_trades,
                SUM(CASE WHEN status = 'SKIPPED' THEN 1 ELSE 0 END) as skipped_trades,
                SUM(COALESCE(realized_pnl_usdc, pnl, 0)) as pnl,
                SUM(CASE WHEN COALESCE(realized_pnl_usdc, pnl, 0) > 0 THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN status = 'SUCCESS' THEN executed_price ELSE NULL END) as avg_price,
                AVG(CASE WHEN status = 'SUCCESS' THEN amount_usdc ELSE NULL END) as avg_bet
            FROM trade_history
            WHERE mode = 'PAPER'
              AND created_at >= NOW() - INTERVAL '24 HOURS'
            GROUP BY asset
            ORDER BY asset;
        """)
        res_asset = (await session.execute(query_asset)).mappings().all()

        # 3. Количество открытых позиций (OPEN)
        query_open = text("""
            SELECT COUNT(*) as open_positions
            FROM trade_history
            WHERE mode = 'PAPER'
              AND position_status = 'OPEN'
              AND status = 'SUCCESS';
        """)
        res_open = (await session.execute(query_open)).scalar() or 0

        print("=== PAPER BASELINE METRICS (LAST 24 HOURS) ===")
        print(f"Total Records (24h): {res_24h['total_trades']}")
        print(f"Status Breakdown: SUCCESS={res_24h['success_trades']}, SKIPPED={res_24h['skipped_trades']}, FAILED={res_24h['failed_trades']}, PENDING={res_24h['pending_trades']}")
        
        success_cnt = res_24h['success_trades'] or 0
        win_cnt = res_24h['winning_trades'] or 0
        wr = (win_cnt / success_cnt * 100) if success_cnt > 0 else 0.0
        
        print(f"Total PnL (24h): {res_24h['total_pnl']:.4f} USDC")
        print(f"Win Rate (24h): {wr:.2f}% ({win_cnt}/{success_cnt})")
        print(f"Avg Executed Price: ${res_24h['avg_executed_price'] or 0:.4f}")
        print(f"Avg Bet Size: ${res_24h['avg_bet_size'] or 0:.4f} USDC")
        print(f"Current Open Positions (PAPER): {res_open}")

        print("\n--- Asset Breakdown (24h) ---")
        for r in res_asset:
            a_success = r['success_trades'] or 0
            a_wins = r['wins'] or 0
            a_wr = (a_wins / a_success * 100) if a_success > 0 else 0.0
            print(f"[{r['asset']}] Total={r['total_trades']}, Executed={a_success}, Skipped={r['skipped_trades']}, PnL={r['pnl']:.4f} USDC, WR={a_wr:.1f}%, AvgPrice=${r['avg_price'] or 0:.3f}, AvgBet=${r['avg_bet'] or 0:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
