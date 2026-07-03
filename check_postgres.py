import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func
from polyflip.db.models import MarketSnapshot, LiveMarket, TradeHistory, CollectorStatus

async def main():
    # Try localhost postgres first
    # Database URL on host might be localhost:5432 if it's forwarded, or we can check
    urls = [
        "postgresql+asyncpg://polyflip:secret@127.0.0.1:5432/polyflip",
    ]
    
    for url in urls:
        print(f"Connecting to {url}...")
        try:
            engine = create_async_engine(url, echo=False)
            async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session() as session:
                # Count snapshots
                stmt = select(func.count(MarketSnapshot.id))
                res = await session.execute(stmt)
                count = res.scalar()
                print(f"Connection success! Total snapshots: {count}")
                
                # Count by outcome
                stmt = select(MarketSnapshot.final_outcome, func.count(MarketSnapshot.id)).group_by(MarketSnapshot.final_outcome)
                res = await session.execute(stmt)
                print("Snapshots by outcome:")
                for row in res.all():
                    print(f"  {row[0]}: {row[1]}")
                    
                # Count trade history
                stmt = select(func.count(TradeHistory.id))
                res = await session.execute(stmt)
                print(f"Total trades in history: {res.scalar()}")
                
                # Count live markets
                stmt = select(func.count(LiveMarket.market_id))
                res = await session.execute(stmt)
                print(f"Total live markets: {res.scalar()}")
                
            await engine.dispose()
            return
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
