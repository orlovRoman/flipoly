import asyncio
from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry
from sqlalchemy import select

async def main():
    async with async_session() as s:
        stmt = select(ModelRegistry.asset, ModelRegistry.version, ModelRegistry.trained_at, ModelRegistry.is_active).order_by(ModelRegistry.trained_at.desc()).limit(30)
        res = await s.execute(stmt)
        for r in res.all():
            print(f"{r.asset:15} | v{r.version:2} | trained_at: {r.trained_at} | active: {r.is_active}")

if __name__ == "__main__":
    asyncio.run(main())
