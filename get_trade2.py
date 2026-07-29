import asyncio
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

async def run():
    async with async_session() as db:
        res = await db.execute(
            select(TradeHistory)
            .where(TradeHistory.position_status == 'OPEN', TradeHistory.mode == 'PAPER')
            .limit(1)
        )
        trade = res.scalar()
        if trade:
            print(f"ID: {trade.id}, market_id: {trade.market_id}, title: {getattr(trade, 'market_title', None)}, condition_id: {getattr(trade, 'condition_id', None)}")

asyncio.run(run())
