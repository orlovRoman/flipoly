import sys
import asyncio
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

async def run():
    async with async_session() as db:
        res = await db.execute(
            select(TradeHistory.market_id)
            .where(TradeHistory.position_status == 'OPEN', TradeHistory.mode == 'PAPER')
            .limit(1)
        )
        print(res.scalar())

asyncio.run(run())
