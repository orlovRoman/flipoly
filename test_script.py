import sys
print("starting script")
import argparse
import asyncio
import aiohttp
from decimal import Decimal
from sqlalchemy import select
print("imports 1")
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
print("imports 2")
from polyflip.execution.settlement_service import settle_resolved_position, AccountingInvariantError
from polyflip.execution.states import ACTIVE_POSITION_STATES
from polyflip.collector.resolver import extract_final_outcome
print("imports 3")

async def main():
    print("main entered")
    async with async_session() as db:
        print("db session created")
        res = await db.execute(select(TradeHistory).limit(1))
        print("db execute done")
        print(res.scalars().all())

asyncio.run(main())
