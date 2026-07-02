import asyncio
from datetime import datetime, timezone
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import MarketSnapshot, ModelRegistry
from polyflip.api.backtest_schemas import BacktestConfig
from polyflip.backtesting.market_replay import group_snapshots_into_replays
from polyflip.backtesting.runner import BacktestRunner

async def main():
    print("Starting debug backtest...")
    config = BacktestConfig(
        assets=["BTC", "ETH", "SOL", "DOGE"],
        strategy_mode="ML"
    )
    
    async with async_session() as db:
        # Load snapshots exactly like backtest_api.py does
        from sqlalchemy import func
        base_filters = [
            MarketSnapshot.asset.in_(config.assets),
            MarketSnapshot.final_outcome.in_(["YES", "NO"]),
            MarketSnapshot.time_left_min >= config.min_time_left_min,
            MarketSnapshot.time_left_min <= config.max_time_left_min,
        ]
        
        count_cte = (
            select(MarketSnapshot.market_id, func.count().label("total_snaps"))
            .where(*base_filters)
            .group_by(MarketSnapshot.market_id)
            .cte("market_counts")
        )

        rank_sub = (
            select(
                MarketSnapshot.id.label("snap_id"),
                func.row_number().over(
                    partition_by=MarketSnapshot.market_id,
                    order_by=MarketSnapshot.time_left_min.desc()
                ).label("rn"),
                count_cte.c.total_snaps,
            )
            .join(count_cte, MarketSnapshot.market_id == count_cte.c.market_id)
            .where(*base_filters)
            .subquery("ranked_snaps")
        )

        qualified_markets = select(rank_sub.c.market_id).where(
            rank_sub.c.rn == 1,
            rank_sub.c.total_snaps >= config.min_snapshots_per_market,
        )

        stmt = select(MarketSnapshot).where(
            MarketSnapshot.market_id.in_(qualified_markets)
        )
        if config.date_from:
            stmt = stmt.where(MarketSnapshot.recorded_at >= config.date_from)
        if config.date_to:
            stmt = stmt.where(MarketSnapshot.recorded_at <= config.date_to)

        result = await db.execute(stmt)
        snapshots = result.scalars().all()
        print(f"Loaded snapshots count: {len(snapshots)}")
        
        replays = group_snapshots_into_replays(snapshots, min_snapshots=1)
        print(f"Grouped into replays count: {len(replays)}")
        
        # Load active model
        model_stmt = select(ModelRegistry).where(ModelRegistry.is_active == True).where(ModelRegistry.asset == "BTC")
        model_row = (await db.execute(model_stmt)).scalars().first()
        if model_row:
            print(f"Found active model for BTC: version={model_row.version}")
            model_blob = model_row.model_blob
            features_str = model_row.features or ""
        else:
            print("No active model found for BTC!")
            model_blob = None
            features_str = ""

        runner_config = config.to_runner_config()
        runner = BacktestRunner(runner_config, model_blob, features_str)
        
        # Trace why run_market might do nothing
        for i, (market_id, replay) in enumerate(list(replays.items())[:5]):
            print(f"\nTracing replay {i}: market_id={market_id}, asset={replay.asset}, snapshots={len(replay.ticks)}")
            min_time = float(runner.config.get("MIN_TIME_LEFT_MIN", 1.0))
            max_time = float(runner.config.get("MAX_TIME_LEFT_MIN", 60.0))
            ticks = replay.get_ticks_in_window(min_time, max_time)
            print(f"Ticks in window [{min_time}, {max_time}]: {len(ticks)}")
            
            for t in ticks[:3]:
                decision, p_flip, signal = runner._evaluate_tick(t)
                print(f"  Tick time_left={t.time_left_min:.2f}: p_flip={p_flip:.4f}, action={decision.action}, reason={decision.reason}")

if __name__ == "__main__":
    asyncio.run(main())
