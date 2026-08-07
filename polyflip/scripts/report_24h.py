import asyncio
import json
from decimal import Decimal
from sqlalchemy import text
from polyflip.db.connection import async_session

def default_json(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

async def main():
    async with async_session() as s:
        # 1. Assets summary
        r1 = await s.execute(text("""
            SELECT 
                asset,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN status = 'SKIPPED' THEN 1 END) as skipped_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                COUNT(CASE WHEN pnl = 0 AND status = 'SUCCESS' THEN 1 END) as zero_pnl_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl_usdc,
                ROUND(CAST(COALESCE(SUM(amount_usdc), 0) AS numeric), 2) as total_volume_usdc
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY asset
            ORDER BY total_pnl_usdc DESC;
        """))
        assets_data = [dict(x) for x in r1.mappings()]

        # 2. Strategy summary
        r2 = await s.execute(text("""
            SELECT 
                COALESCE(strategy_type, 'N/A') as strategy,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN status = 'SKIPPED' THEN 1 END) as skipped_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl_usdc
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY COALESCE(strategy_type, 'N/A')
            ORDER BY total_pnl_usdc DESC;
        """))
        strategies_data = [dict(x) for x in r2.mappings()]

        # 3. Matrix (Asset x Strategy)
        r3 = await s.execute(text("""
            SELECT 
                asset,
                COALESCE(strategy_type, 'N/A') as strategy,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl_usdc
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY asset, COALESCE(strategy_type, 'N/A')
            ORDER BY asset, total_pnl_usdc DESC;
        """))
        matrix_data = [dict(x) for x in r3.mappings()]

        # 4. Overall Totals
        r4 = await s.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN status = 'SKIPPED' THEN 1 END) as skipped_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl_usdc,
                ROUND(CAST(COALESCE(SUM(amount_usdc), 0) AS numeric), 2) as total_volume_usdc
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours';
        """))
        overall_data = [dict(x) for x in r4.mappings()][0]

        report = {
            "overall": overall_data,
            "assets": assets_data,
            "strategies": strategies_data,
            "matrix": matrix_data
        }
        print(json.dumps(report, default=default_json, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
