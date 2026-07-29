import asyncio
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

async def run():
    async with async_session() as db:
        res = await db.execute(
            select(TradeHistory)
            .where(TradeHistory.id == 18477)
            .limit(1)
        )
        trade = res.scalar()
        if trade:
            print(f"ID: {trade.id}, market_id: {trade.market_id}, asset: {trade.asset}")

asyncio.run(run())
