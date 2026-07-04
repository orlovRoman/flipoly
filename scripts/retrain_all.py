import asyncio
import structlog
from polyflip.db.connection import async_session
from polyflip.crypto.trainer import CryptoModelTrainer

logger = structlog.get_logger(__name__)

async def main():
    async with async_session() as db:
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            t = CryptoModelTrainer(db)
            try:
                ok = await t.train(symbol, "15m")
                print(f"Retrained {symbol}: {ok}")
            except Exception as e:
                print(f"Failed to retrain {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
