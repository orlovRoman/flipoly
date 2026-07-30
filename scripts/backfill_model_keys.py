"""
scripts/backfill_model_keys.py
По-настоящему пакетный, валидируемый backfill с ключами keyset-пагинации (id > last_id).

Запуск:
  python -m scripts.backfill_model_keys --dry-run
  python -m scripts.backfill_model_keys --apply --mode PAPER
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update, func, cast, Numeric
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory, DecisionFunnelLog, ModelRegistry

BATCH_SIZE = 500

async def run_backfill(
    apply_changes: bool = False,
    mode: str = "PAPER",
    before_str: str = None,
    after_id: int = 0,
    session_override = None,
):
    print(f"=== Starting Truly Batched Backfill (mode='{mode}', dry_run={not apply_changes}, after_id={after_id}) ===")

    before_dt = None
    if before_str:
        try:
            before_dt = datetime.fromisoformat(before_str).astimezone(timezone.utc)
            print(f"Cutoff created_at < {before_dt.isoformat()}")
        except Exception as e:
            print(f"Invalid --before datetime format: {e}")
            return

    stats = {
        "RECONSTRUCTED": 0,
        "AMBIGUOUS": 0,
        "INVALID_KEY": 0,
        "CONFIRM_RECONSTRUCTED": 0,
        "PROCESSED": 0,
    }

    if session_override is not None:
        await _execute_backfill_logic(session_override, apply_changes, mode, before_dt, after_id, stats)
    else:
        async with async_session() as session:
            await _execute_backfill_logic(session, apply_changes, mode, before_dt, after_id, stats)

async def _execute_backfill_logic(session, apply_changes, mode, before_dt, after_id, stats):
        # 1. Загружаем список валидных моделей из реестра
        registry_rows = (await session.execute(select(ModelRegistry.asset, ModelRegistry.version))).all()
        valid_models = {(row.asset, row.version) for row in registry_rows}
        print(f"Loaded {len(valid_models)} valid (asset, version) entries from ModelRegistry.")

        pnl_expr = func.coalesce(TradeHistory.realized_pnl_usdc, cast(TradeHistory.pnl, Numeric))
        last_id = after_id

        while True:
            # Строим условие выборок строго частями по BATCH_SIZE
            conditions = [
                TradeHistory.id > last_id,
                TradeHistory.mode == mode,
                TradeHistory.position_status == "CLOSED",
                TradeHistory.model_key.is_(None),
                TradeHistory.model_version.is_not(None),
                pnl_expr.is_not(None),
            ]
            if before_dt:
                conditions.append(TradeHistory.created_at < before_dt)

            stmt = (
                select(TradeHistory)
                .where(*conditions)
                .order_by(TradeHistory.id.asc())
                .limit(BATCH_SIZE)
            )

            batch = (await session.execute(stmt)).scalars().all()
            if not batch:
                break

            print(f"Processing batch of {len(batch)} trades (IDs {batch[0].id} to {batch[-1].id})...")

            for trade in batch:
                stats["PROCESSED"] += 1
                found_key = None
                confirm_key = None
                confirm_version = None

                # Извлекаем метаданные LightGBM
                if trade.lgbm_metadata:
                    try:
                        meta = json.loads(trade.lgbm_metadata)
                        if isinstance(meta, dict):
                            if meta.get("ml_phase_model"):
                                found_key = meta["ml_phase_model"]
                            c_key = meta.get("lgbm_model_key")
                            c_ver = meta.get("lgbm_version")
                            if c_key and c_ver is not None:
                                try:
                                    c_ver_int = int(c_ver)
                                    if (c_key, c_ver_int) in valid_models:
                                        confirm_key = c_key
                                        confirm_version = c_ver_int
                                        stats["CONFIRM_RECONSTRUCTED"] += 1
                                except ValueError:
                                    pass
                    except Exception:
                        pass

                # Если model_key не найден в metadata, сопоставляем с decision_funnel_log
                if not found_key and trade.created_at and trade.market_id:
                    window_start = trade.created_at - timedelta(seconds=30)
                    window_end = trade.created_at + timedelta(seconds=30)
                    expected_action = f"BUY_{trade.outcome_bought}" if trade.outcome_bought in ("YES", "NO") else None

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

                # Валидируем найденный ключ по реестру моделей
                if found_key:
                    if (found_key, trade.model_version) in valid_models:
                        if apply_changes:
                            trade.model_key = found_key
                            trade.model_attribution_source = "RECONSTRUCTED"
                        stats["RECONSTRUCTED"] += 1
                    else:
                        if apply_changes:
                            trade.model_key = None
                            trade.model_attribution_source = "AMBIGUOUS"
                        stats["INVALID_KEY"] += 1
                else:
                    if apply_changes:
                        trade.model_key = None
                        trade.model_attribution_source = "AMBIGUOUS"
                    stats["AMBIGUOUS"] += 1

                if confirm_key and confirm_version is not None:
                    if apply_changes:
                        trade.confirm_model_key = confirm_key
                        trade.confirm_model_version = confirm_version

            last_id = batch[-1].id

            if apply_changes:
                await session.commit()
                print(f"  Committed batch up to ID {last_id}")

        print("\n=== Backfill Summary Report ===")
        print(f"  Processed trades:       {stats['PROCESSED']}")
        print(f"  Reconstructed model_key:{stats['RECONSTRUCTED']}")
        print(f"  Ambiguous attribution:  {stats['AMBIGUOUS']}")
        print(f"  Invalid model keys:     {stats['INVALID_KEY']}")
        print(f"  Reconstructed confirms: {stats['CONFIRM_RECONSTRUCTED']}")

        if not apply_changes:
            print("\nDry-run complete. No database changes committed.")
        else:
            print("\n=== Apply Completed Successfully ===")

def main():
    parser = argparse.ArgumentParser(description="Truly batched model attribution backfill script")
    parser.add_argument("--apply", action="store_true", help="Apply changes to database")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without committing")
    parser.add_argument("--mode", type=str, default="PAPER", help="Trading mode (default: PAPER)")
    parser.add_argument("--before", type=str, default=None, help="ISO datetime cutoff for created_at")
    parser.add_argument("--after-id", type=int, default=0, help="Start after specified trade ID")
    args = parser.parse_args()

    apply_changes = args.apply and not args.dry_run
    asyncio.run(run_backfill(
        apply_changes=apply_changes,
        mode=args.mode,
        before_str=args.before,
        after_id=args.after_id,
    ))

if __name__ == "__main__":
    main()
