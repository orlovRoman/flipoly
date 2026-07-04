import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from polyflip.db.models import Base
from polyflip.crypto.candle_collector import collect_new_candles

async def run():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as session:
        # При пустой БД должен запросить 24h истории без ошибок
        results = await collect_new_candles(session)

    for symbol, count in results.items():
        print(f"  {symbol}: {count} новых свечей")
    print("✅ Инкрементальный коллектор работает без исключений")

asyncio.run(run())
