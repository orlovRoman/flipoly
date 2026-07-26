import argparse
import asyncio
import os
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
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Please specify either --dry-run or --apply")
        return

    # Database connection
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://polyflip:secret@localhost/polyflip")
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get outcome distribution
        result = await session.execute(text("SELECT final_outcome, COUNT(*) FROM market_snapshots GROUP BY final_outcome"))
        outcomes = result.fetchall()
        
        print("--- Current Database Outcomes ---")
        for outcome, count in outcomes:
            print(f"{outcome}: {count}")
        print("---------------------------------")
        
        # Find invalid outcomes
        valid_outcomes = {"YES", "NO", "INVALID", "PENDING"}
        invalid_outcomes = [o for o, c in outcomes if o not in valid_outcomes]
        
        if not invalid_outcomes:
            print("No invalid outcomes found! All good.")
            return
            
        print(f"Found invalid outcomes: {invalid_outcomes}")
        
        # We need to fetch the unique market_ids for these invalid outcomes
        # and re-resolve them using the new extract_final_outcome logic
        
        # Getting unique market IDs with invalid outcomes
        invalid_markets_res = await session.execute(
            text(f"SELECT DISTINCT market_id FROM market_snapshots WHERE final_outcome NOT IN ('YES', 'NO', 'INVALID', 'PENDING')")
        )
        market_ids = [row[0] for row in invalid_markets_res.fetchall()]
        print(f"Total markets affected: {len(market_ids)}")
        
        if args.dry_run:
            print("Run with --apply to fetch Gamma API and resolve these markets.")
            return
            
        print("Applying fixes...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            for market_id in market_ids:
                try:
                    response = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
                    await asyncio.sleep(0.2)
                    
                    if response.status_code != 200:
                        print(f"Failed to fetch market {market_id}")
                        continue
                        
                    market_data = response.json()
                    new_outcome = extract_final_outcome(market_data)
                    
                    if new_outcome is None:
                        print(f"Market {market_id} outcome still could not be determined. Setting to PENDING.")
                        new_outcome = "PENDING"
                        
                    # Update all snapshots for this market
                    print(f"Market {market_id}: -> {new_outcome}")
                    
                    await session.execute(
                        text("UPDATE market_snapshots SET final_outcome = :outcome WHERE market_id = :market_id"),
                        {"outcome": new_outcome, "market_id": market_id}
                    )
                    
                    # Also need to re-calculate flip_vs_final if YES/NO
                    if new_outcome in ("YES", "NO"):
                        # Get all snapshots to recalculate
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
                        
                except Exception as e:
                    print(f"Error processing {market_id}: {e}")
                    
            await session.commit()
            print("Fixes committed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
