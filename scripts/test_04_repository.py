import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from polyflip.db.models import Base, CryptoCandle
from polyflip.crypto.candle_repository import upsert_candles, get_recent_candles, get_latest_open_time

async def run():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, expire_on_commit=False)

    candle = {
        "open_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        "open": 40000.0, "high": 41000.0, "low": 39500.0,
        "close": 40500.0, "volume": 123.5, "taker_buy_volume": 60.0,
    }

    async with Session() as s:
        # SQLite не поддерживает pg_insert, тестируем через ORM напрямую
        obj = CryptoCandle(symbol="BTCUSDT", interval="15m", source="binance", **candle)
        s.add(obj)
        await s.commit()
        rows = await get_recent_candles(s, "BTCUSDT", "15m", limit=10)
        assert len(rows) == 1
        latest = await get_latest_open_time(s, "BTCUSDT", "15m")
        if latest and latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        assert latest == candle["open_time"], f"{latest} != {candle['open_time']}"

    print("✅ CandleRepository OK — get_recent / get_latest работают")

asyncio.run(run())
