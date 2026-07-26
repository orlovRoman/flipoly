import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import os

DATABASE_URL = "postgresql+asyncpg://polyflip:secret@127.0.0.1:5432/polyflip"

async def fix():
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        await session.execute(text("UPDATE trade_history SET status = 'SUCCESS' WHERE status = 'PENDING' AND id IN (SELECT trade_history_id FROM execution_requests WHERE state = 'FILLED')"))
        await session.execute(text("UPDATE trade_history SET status = 'FAILED' WHERE status = 'PENDING' AND id IN (SELECT trade_history_id FROM execution_requests WHERE state IN ('FAILED', 'REJECTED'))"))
        await session.commit()
    print("Done")

if __name__ == "__main__":
    asyncio.run(fix())
