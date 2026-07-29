import sys
import asyncio

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

print_flush("Script started")

from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from polyflip.execution.states import ACTIVE_POSITION_STATES

print_flush("Imports completed")

async def test_db():
    print_flush("Connecting to DB...")
    try:
        async with async_session() as db:
            print_flush("Session created, executing query...")
            query = select(TradeHistory).where(
                TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
                TradeHistory.mode.in_(("PAPER", "SHADOW")),
            )
            res = await asyncio.wait_for(db.execute(query), timeout=5.0)
            trades = res.scalars().all()
            print_flush(f"Query result: {len(trades)} trades found")
    except Exception as e:
        print_flush(f"Error: {e}")

asyncio.run(test_db())
print_flush("Script finished")
