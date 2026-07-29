import sys
import asyncio

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

print_flush("Script started")

from sqlalchemy import text
from polyflip.db.connection import async_session

print_flush("Imports completed")

async def test_db():
    print_flush("Connecting to DB...")
    try:
        async with async_session() as db:
            print_flush("Session created, executing query...")
            res = await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=5.0)
            print_flush(f"Query result: {res.scalar()}")
    except Exception as e:
        print_flush(f"Error: {e}")

asyncio.run(test_db())
print_flush("Script finished")
