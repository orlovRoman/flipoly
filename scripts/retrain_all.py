import asyncio
import structlog
from polyflip.db.connection import async_session
from polyflip.crypto.trainer import CryptoModelTrainer

logger = structlog.get_logger(__name__)

async def main():
    async with async_session() as db:
        from sqlalchemy import select, func
        from polyflip.db.models import CryptoCandle
        counts = (await db.execute(
            select(CryptoCandle.symbol, CryptoCandle.interval, func.count())
            .group_by(CryptoCandle.symbol, CryptoCandle.interval)
        )).all()
        print(f"Candles counts in DB: {counts}")

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            t = CryptoModelTrainer(db)
            try:
                ok = await t.train(symbol, "15m")
                print(f"Retrained {symbol}: {ok}")
            except Exception as e:
                print(f"Failed to retrain {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
