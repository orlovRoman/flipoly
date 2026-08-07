import asyncio
import json
from decimal import Decimal
from datetime import datetime
from sqlalchemy import text
from polyflip.db.connection import async_session

def default_json(obj):
    if isinstance(obj, (Decimal, float)):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

async def main():
    async with async_session() as s:
        # 1. Fetch all models in ModelRegistry
        m_stmt = text("""
            SELECT 
                id, asset, version, is_active, accuracy, baseline,
                decision_threshold, decision_threshold_down,
                train_samples, validation_samples, positive_rate,
                precision_at_threshold, recall_at_threshold, f1_at_threshold, brier_score, ece,
                backtest_pnl, backtest_trades, backtest_wr,
                quality_gate_passed, quality_gate_reasons,
                activation_source, activated_at, trained_at, training_params, features
            FROM model_registry
            ORDER BY asset, version DESC;
        """)
        models_res = await s.execute(m_stmt)
        models_data = [dict(x) for x in models_res.mappings()]

        # 2. Realized trading stats per model version from trade_history
        t_stmt = text("""
            SELECT 
                asset,
                COALESCE(CAST(model_version AS text), 'N/A') as model_version,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl_usdc,
                ROUND(CAST(COALESCE(SUM(amount_usdc), 0) AS numeric), 2) as total_volume_usdc
            FROM trade_history
            GROUP BY asset, COALESCE(CAST(model_version AS text), 'N/A')
            ORDER BY asset, total_pnl_usdc DESC;
        """)
        trades_res = await s.execute(t_stmt)
        trades_data = [dict(x) for x in trades_res.mappings()]

        # 3. 24h Realized trading stats per model version
        t24_stmt = text("""
            SELECT 
                asset,
                COALESCE(CAST(model_version AS text), 'N/A') as model_version,
                COUNT(*) as total_records,
                COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as win_count,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as loss_count,
                ROUND(CAST(COALESCE(SUM(pnl), 0) AS numeric), 2) as total_pnl_24h
            FROM trade_history
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY asset, COALESCE(CAST(model_version AS text), 'N/A')
            ORDER BY asset, total_pnl_24h DESC;
        """)
        trades24_res = await s.execute(t24_stmt)
        trades24_data = [dict(x) for x in trades24_res.mappings()]

        out = {
            "models": models_data,
            "all_time_trades_by_model": trades_data,
            "trades_24h_by_model": trades24_data
        }
        print(json.dumps(out, default=default_json, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
