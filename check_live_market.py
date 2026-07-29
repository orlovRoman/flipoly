import asyncio
from sqlalchemy import select, desc
from polyflip.db.connection import async_session
from polyflip.db.models import LiveMarket

async def run():
    async with async_session() as db:
        res = await db.execute(
            select(LiveMarket)
            .limit(2)
        )
        markets = res.scalars().all()
        for m in markets:
            print(f"market_id: {m.market_id}, yes_token_id: {m.yes_token_id}")

asyncio.run(run())
