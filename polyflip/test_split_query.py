import asyncio
import time
from sqlalchemy import select, func
from polyflip.db.connection import async_session
from polyflip.db.models import MarketSnapshot
from polyflip.api.backtest_schemas import BacktestConfig

async def main():
    print("Testing split query approach...")
    config = BacktestConfig(
        assets=["BTC", "ETH", "SOL", "DOGE"],
        strategy_mode="ML"
    )
    
    async with async_session() as db:
        start_t = time.time()
        
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
                MarketSnapshot.market_id.label("market_id"),
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

        # Query 1: Get qualified market ids
        q1_start = time.time()
        qualified_markets_stmt = select(rank_sub.c.market_id).where(
            rank_sub.c.rn == 1,
            rank_sub.c.total_snaps >= config.min_snapshots_per_market,
        )
        res1 = await db.execute(qualified_markets_stmt)
        market_ids = [row[0] for row in res1.all()]
        q1_end = time.time()
        print(f"Query 1 (find market_ids) took: {q1_end - q1_start:.4f}s. Found {len(market_ids)} markets.")

        if not market_ids:
            print("No qualified markets found!")
            return

        # Query 2: Get all snapshots for those market ids
        q2_start = time.time()
        stmt = select(MarketSnapshot).where(
            MarketSnapshot.market_id.in_(market_ids)
        )
        if config.date_from:
            stmt = stmt.where(MarketSnapshot.recorded_at >= config.date_from)
        if config.date_to:
            stmt = stmt.where(MarketSnapshot.recorded_at <= config.date_to)
            
        res2 = await db.execute(stmt)
        snapshots = res2.scalars().all()
        q2_end = time.time()
        print(f"Query 2 (fetch snapshots) took: {q2_end - q2_start:.4f}s. Loaded {len(snapshots)} snapshots.")
        print(f"Total query execution time: {time.time() - start_t:.4f}s.")

if __name__ == "__main__":
    asyncio.run(main())
