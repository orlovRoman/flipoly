import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        res = await conn.execute(text("SELECT key, value FROM runtime_settings WHERE value = 'None'"))
        rows = res.fetchall()
        print("BAD ROWS:", rows)
        
        # fix them
        if rows:
            await conn.execute(text("DELETE FROM runtime_settings WHERE value = 'None'"))
            print("Deleted bad rows.")

if __name__ == "__main__":
    asyncio.run(main())
