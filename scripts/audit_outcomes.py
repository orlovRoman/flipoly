import argparse
import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from polyflip.db.models import MarketSnapshot
import structlog
import httpx
from polyflip.collector.resolver import extract_final_outcome

logger = structlog.get_logger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Audit and fix MarketSnapshot final outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    parser.add_argument("--apply", action="store_true", help="Apply fixes to the database")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing markets")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Please specify either --dry-run or --apply")
        return

    # Database connection
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
        sys.exit(1)
        
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            # Get outcome distribution
            result = await session.execute(text("SELECT final_outcome, COUNT(*) FROM market_snapshots GROUP BY final_outcome"))
            outcomes = result.fetchall()
            
            print("--- Current Database Outcomes ---")
            for outcome, count in outcomes:
                print(f"{outcome}: {count}")
            print("---------------------------------")
            
            valid_outcomes = {"YES", "NO", "INVALID", "PENDING"}
            invalid_outcomes = [o for o, c in outcomes if o not in valid_outcomes]
            
            if not invalid_outcomes:
                print("No invalid outcomes found! All good.")
                return
                
            print(f"Found invalid outcomes: {invalid_outcomes}")
            
            invalid_markets_res = await session.execute(
                text("SELECT DISTINCT market_id FROM market_snapshots WHERE final_outcome NOT IN ('YES', 'NO', 'INVALID', 'PENDING')")
            )
            market_ids = [row[0] for row in invalid_markets_res.fetchall()]
            total_markets = len(market_ids)
            print(f"Total markets affected: {total_markets}")
            
            if args.dry_run:
                print("Run with --apply to fetch Gamma API and resolve these markets.")
                return
                
            print("Applying fixes...")
            success_count = 0
            error_count = 0
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for i in range(0, total_markets, args.batch_size):
                    batch = market_ids[i:i + args.batch_size]
                    
                    for market_id in batch:
                        try:
                            # Using savepoint for each market
                            async with session.begin_nested():
                                response = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
                                await asyncio.sleep(0.2)
                                
                                if response.status_code != 200:
                                    print(f"Failed to fetch market {market_id}")
                                    error_count += 1
                                    continue
                                    
                                market_data = response.json()
                                new_outcome = extract_final_outcome(market_data)
                                
                                if new_outcome is None:
                                    print(f"Market {market_id} outcome still could not be determined. Setting to PENDING.")
                                    new_outcome = "PENDING"
                                    
                                print(f"Market {market_id}: -> {new_outcome}")
                                
                                await session.execute(
                                    text("UPDATE market_snapshots SET final_outcome = :outcome WHERE market_id = :market_id"),
                                    {"outcome": new_outcome, "market_id": market_id}
                                )
                                
                                if new_outcome in ("YES", "NO"):
                                    snaps_res = await session.execute(
                                        text("SELECT id, mid_price FROM market_snapshots WHERE market_id = :market_id"),
                                        {"market_id": market_id}
                                    )
                                    for snap_id, mid_price in snaps_res.fetchall():
                                        if mid_price == 0.5:
                                            flip = False
                                        else:
                                            market_believed_yes = mid_price > 0.5
                                            actual_is_yes = (new_outcome == "YES")
                                            flip = (market_believed_yes != actual_is_yes)
                                        await session.execute(
                                            text("UPDATE market_snapshots SET flip_vs_final = :flip WHERE id = :id"),
                                            {"flip": flip, "id": snap_id}
                                        )
                                else:
                                    await session.execute(
                                        text("UPDATE market_snapshots SET flip_vs_final = NULL WHERE market_id = :market_id"),
                                        {"market_id": market_id}
                                    )
                            success_count += 1
                        except Exception as e:
                            print(f"Error processing {market_id}: {e}")
                            error_count += 1
                            
                    await session.commit()
                    print(f"Committed batch of {len(batch)} markets.")
                    
            print(f"--- Audit Complete ---")
            print(f"Successfully processed: {success_count}")
            print(f"Errors: {error_count}")

    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
