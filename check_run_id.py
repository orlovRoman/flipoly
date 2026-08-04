import asyncio, os, sys
sys.path.insert(0, '.')
from polyflip.db.session import async_session_maker
from polyflip.db.models import TradeHistory
from sqlalchemy import select

async def main():
    async with async_session_maker() as s:
        res = await s.execute(select(TradeHistory.decision_run_id).limit(10).order_by(TradeHistory.id.desc()))
        print([r[0] for r in res])

if __name__ == "__main__":
    asyncio.run(main())
