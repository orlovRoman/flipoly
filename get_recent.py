import asyncio
from sqlalchemy import select, desc
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

async def run():
    async with async_session() as db:
        res = await db.execute(
            select(TradeHistory)
            .order_by(desc(TradeHistory.id))
            .limit(5)
        )
        trades = res.scalars().all()
        for t in trades:
            print(f"ID: {t.id}, market_id: {t.market_id}")

asyncio.run(run())
