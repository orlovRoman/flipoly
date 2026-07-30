"""
scripts/backfill_model_keys.py
Пакетный скрипт безопасной гидратации исторических полей атрибуции моделей для trade_history.
Категоризирует сделки на:
  - EXACT (записанные с точной моделью)
  - RECONSTRUCTED (достоверно восстановленные из decision_funnel_log / lgbm_metadata)
  - AMBIGUOUS (неоднозначные прошлые сделки)
"""

import asyncio
import json
from sqlalchemy import select, update, and_
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory, DecisionFunnelLog

async def backfill():
    async with async_session() as session:
        # 1. Сделки с уже точной моделью
        await session.execute(
            update(TradeHistory)
            .where(TradeHistory.model_key.is_not(None))
            .where(TradeHistory.model_attribution_source.is_(None))
            .values(model_attribution_source="EXACT")
        )
        
        # 2. Необраработанные сделки
        stmt = select(TradeHistory).where(TradeHistory.model_key.is_(None))
        trades = (await session.execute(stmt)).scalars().all()
        
        reconstructed_count = 0
        ambiguous_count = 0
        
        for t in trades:
            found_key = None
            
            # Попытка 1: lgbm_metadata JSON
            if t.lgbm_metadata:
                try:
                    meta = json.loads(t.lgbm_metadata)
                    if isinstance(meta, dict) and meta.get("ml_phase_model"):
                        found_key = meta["ml_phase_model"]
                except Exception:
                    pass
            
            # Попытка 2: decision_funnel_log по market_id
            if not found_key and t.market_id:
                funnel_stmt = (
                    select(DecisionFunnelLog.used_model)
                    .where(DecisionFunnelLog.market_id == t.market_id)
                    .where(DecisionFunnelLog.used_model.is_not(None))
                    .distinct()
                )
                funnel_rows = (await session.execute(funnel_stmt)).scalars().all()
                if len(funnel_rows) == 1:
                    found_key = funnel_rows[0]
            
            if found_key:
                t.model_key = found_key
                t.model_attribution_source = "RECONSTRUCTED"
                reconstructed_count += 1
            else:
                t.model_attribution_source = "AMBIGUOUS"
                ambiguous_count += 1
                
        await session.commit()
        print(f"Backfill Complete! Reconstructed: {reconstructed_count}, Ambiguous: {ambiguous_count}")

if __name__ == "__main__":
    asyncio.run(backfill())
