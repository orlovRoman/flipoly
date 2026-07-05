import asyncio
import structlog
from polyflip.db.connection import async_session
from polyflip.crypto.trainer import CryptoModelTrainer

logger = structlog.get_logger(__name__)

async def main():
    async with async_session() as db:
        from sqlalchemy import select
        from polyflip.db.models import ModelRegistry, RuntimeSettings
        
        models = (await db.execute(select(ModelRegistry))).scalars().all()
        print("MODEL REGISTRY:")
        for m in models:
            print(f"Asset: {m.asset}, Version: {m.version}, IsActive: {m.is_active}, Acc: {m.accuracy}, ECE: {m.ece}")
            
        settings = (await db.execute(select(RuntimeSettings))).scalars().all()
        print("\nRUNTIME SETTINGS:")
        for s in settings:
            if "CRYPTO" in s.key:
                print(f"Key: {s.key}, Value: {s.value}")

if __name__ == "__main__":
    asyncio.run(main())
