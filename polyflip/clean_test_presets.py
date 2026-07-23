import asyncio
from polyflip.db.connection import async_session
from sqlalchemy import text

async def main():
    async with async_session() as session:
        res = await session.execute(text("DELETE FROM config_presets WHERE created_by IN ('pytest', 'system_ath') OR name LIKE '%11-43%';"))
        await session.commit()
        print(f"Deleted test presets count: {res.rowcount}")

if __name__ == "__main__":
    asyncio.run(main())
