import asyncio
from polyflip.db.connection import async_session
from polyflip.models.trainer import ModelTrainer

async def test_xrp():
    async with async_session() as session:
        trainer = ModelTrainer(session)
        res = await trainer.train_model("XRP")
        print(f"XRP train result: {res}")
        print(f"Status message: {trainer.status_messages.get('XRP')}")

if __name__ == "__main__":
    asyncio.run(test_xrp())
