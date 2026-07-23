import asyncio
from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry
from sqlalchemy import select

async def main():
    async with async_session() as s:
        stmt = select(ModelRegistry.asset, ModelRegistry.version, ModelRegistry.trained_at, ModelRegistry.is_active).where(ModelRegistry.asset.like("BTC%")).order_by(ModelRegistry.version.desc())
        res = await s.execute(stmt)
        rows = res.all()
        print(f"Total BTC models found in DB: {len(rows)}")
        for r in rows[:15]:
            print(f"{r.asset:15} | v{r.version:2} | trained_at: {r.trained_at} | active: {r.is_active}")

if __name__ == "__main__":
    asyncio.run(main())
