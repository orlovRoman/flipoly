"""
scripts/backfill_model_keys.py
Пакетный, безопасный backfill с поддержкой --dry-run и --apply.

Запуск:
  python -m scripts.backfill_model_keys --dry-run
  python -m scripts.backfill_model_keys --apply
"""

import argparse
import asyncio
import json
from datetime import timedelta
from sqlalchemy import select, update, func, and_
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory, DecisionFunnelLog

BATCH_SIZE = 500

async def run_backfill(apply_changes: bool = False):
    print(f"=== Starting Model Attribution Backfill (mode={'APPLY' if apply_changes else 'DRY-RUN'}) ===")

    stats = {
        "EXACT": 0,
        "RECONSTRUCTED": 0,
        "AMBIGUOUS": 0,
        "IGNORED": 0,
    }

    async with async_session() as session:
        # 1. Помечаем сделки, которые уже были записаны с точной моделью
        exact_stmt = (
            update(TradeHistory)
            .where(TradeHistory.model_key.is_not(None))
            .where(TradeHistory.model_attribution_source.is_(None))
        )
        if apply_changes:
            res = await session.execute(exact_stmt.values(model_attribution_source="EXACT"))
            await session.commit()
            print(f"Updated existing EXACT rows: {res.rowcount}")

        # Считаем текущий статус
        count_exact_stmt = select(func.count(TradeHistory.id)).where(TradeHistory.model_attribution_source == "EXACT")
        stats["EXACT"] = (await session.execute(count_exact_stmt)).scalar() or 0

        # 2. Выбираем необработанные сделки
        unattributed_stmt = (
            select(TradeHistory)
            .where(TradeHistory.model_key.is_(None))
            .order_by(TradeHistory.id.asc())
        )
        all_unattributed = (await session.execute(unattributed_stmt)).scalars().all()
        print(f"Found {len(all_unattributed)} unattributed trade records to process...")

        batch_updates = []
        
        for trade in all_unattributed:
            # Игнорируем сделки без базовой информации или без версии
            if not trade.market_id or trade.model_version is None or trade.status == "SKIPPED":
                stats["IGNORED"] += 1
                continue

            expected_action = f"BUY_{trade.outcome_bought}" if trade.outcome_bought in ("YES", "NO") else None
            found_key = None

            # Попытка 1: извлечение из lgbm_metadata (для COMBINED)
            if trade.lgbm_metadata:
                try:
                    meta = json.loads(trade.lgbm_metadata)
                    if isinstance(meta, dict) and meta.get("ml_phase_model"):
                        found_key = meta["ml_phase_model"]
                except Exception:
                    pass

            # Попытка 2: точное сопоставление с decision_funnel_log
            if not found_key and trade.created_at:
                window_start = trade.created_at - timedelta(seconds=30)
                window_end = trade.created_at + timedelta(seconds=30)

                funnel_stmt = (
                    select(DecisionFunnelLog.used_model)
                    .where(DecisionFunnelLog.market_id == trade.market_id)
                    .where(DecisionFunnelLog.asset == trade.asset)
                    .where(DecisionFunnelLog.created_at >= window_start)
                    .where(DecisionFunnelLog.created_at <= window_end)
                    .where(DecisionFunnelLog.used_model.is_not(None))
                )
                if expected_action:
                    funnel_stmt = funnel_stmt.where(DecisionFunnelLog.final_action == expected_action)

                funnel_models = (await session.execute(funnel_stmt)).scalars().all()
                distinct_models = set(funnel_models)

                if len(distinct_models) == 1:
                    found_key = list(distinct_models)[0]

            if found_key:
                batch_updates.append((trade.id, found_key, "RECONSTRUCTED"))
                stats["RECONSTRUCTED"] += 1
            else:
                batch_updates.append((trade.id, None, "AMBIGUOUS"))
                stats["AMBIGUOUS"] += 1

        print("\n=== Backfill Summary Report ===")
        print(f"  EXACT:         {stats['EXACT']}")
        print(f"  RECONSTRUCTED: {stats['RECONSTRUCTED']}")
        print(f"  AMBIGUOUS:     {stats['AMBIGUOUS']}")
        print(f"  IGNORED:       {stats['IGNORED']}")

        if apply_changes and batch_updates:
            print(f"\nApplying {len(batch_updates)} updates in batches of {BATCH_SIZE}...")
            for i in range(0, len(batch_updates), BATCH_SIZE):
                chunk = batch_updates[i:i + BATCH_SIZE]
                for trade_id, m_key, attr_src in chunk:
                    await session.execute(
                        update(TradeHistory)
                        .where(TradeHistory.id == trade_id)
                        .values(model_key=m_key, model_attribution_source=attr_src)
                    )
                await session.commit()
                print(f"  Committed batch {i // BATCH_SIZE + 1} ({len(chunk)} rows)")
            print("=== Apply Completed Successfully ===")
        else:
            print("\nDry-run complete. No changes were committed to database.")

def main():
    parser = argparse.ArgumentParser(description="Backfill model attribution keys in trade_history")
    parser.add_argument("--apply", action="store_true", help="Apply changes to the database")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run without writing")
    args = parser.parse_args()

    apply_changes = args.apply and not args.dry_run
    asyncio.run(run_backfill(apply_changes=apply_changes))

if __name__ == "__main__":
    main()
