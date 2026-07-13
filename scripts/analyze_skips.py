import asyncio
from collections import Counter
from sqlalchemy import select, desc
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from datetime import datetime, timedelta, timezone

async def analyze():
    print("Connecting to database...")
    async with async_session() as session:
        # Get all skipped trades from the last 24 hours
        time_threshold = datetime.now(timezone.utc) - timedelta(days=1)
        
        stmt = select(TradeHistory).where(
            TradeHistory.status == 'SKIP'
        ).where(
            TradeHistory.created_at >= time_threshold
        ).order_by(desc(TradeHistory.created_at))
        
        result = await session.execute(stmt)
        trades = result.scalars().all()

    if not trades:
        print("No skipped trades found in the last 24 hours.")
        return

    reasons = Counter([t.error_msg for t in trades])
    print(f"\n=== ANALYSIS OF SKIPPED OUTSIDER TRADES (Last 24h) ===")
    print(f"Total evaluated opportunities: {len(trades)}")
    print("\nTop reasons why trades were blocked:")
    for reason, count in reasons.most_common():
        pct = (count / len(trades)) * 100
        print(f" [{pct:5.1f}%] {count:4d} times : {reason}")

if __name__ == "__main__":
    asyncio.run(analyze())
