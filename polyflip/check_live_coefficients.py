import asyncio
from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry
from polyflip.api.analytics import extract_coefficients_from_blob
from sqlalchemy import select

async def main():
    async with async_session() as s:
        stmt = select(ModelRegistry).where(ModelRegistry.is_active).limit(5)
        models = (await s.execute(stmt)).scalars().all()
        for m in models:
            coefs = extract_coefficients_from_blob(m.model_blob, m.features)
            print(f"Asset: {m.asset:15} | v{m.version:2} | Extracted coefs count: {len(coefs)}")

if __name__ == "__main__":
    asyncio.run(main())
