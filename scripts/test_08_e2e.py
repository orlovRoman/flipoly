"""E2E: Binance API → SQLite → feature_builder. Один реальный запрос к Binance."""
import asyncio
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from polyflip.db.models import Base, CryptoCandle
from polyflip.crypto.binance_client import fetch_klines
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

async def run():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    print("1️⃣  Запрос к Binance...")
    candles = fetch_klines("BTCUSDT", "15m", limit=200)
    print(f"   Получено: {len(candles)} свечей")

    print("2️⃣  Сохраняем в SQLite...")
    async with Session() as s:
        for c in candles:
            obj = CryptoCandle(
                symbol="BTCUSDT", interval="15m", source="binance",
                open_time=c["open_time"], open=c["open"], high=c["high"],
                low=c["low"], close=c["close"], volume=c["volume"],
                taker_buy_volume=c["taker_buy_volume"],
            )
            s.add(obj)
        await s.commit()

    print("3️⃣  Строим фичи...")
    async with Session() as s:
        rows = await get_recent_candles(s, "BTCUSDT", "15m", limit=200)

    result = build_crypto_features(rows)
    print(f"   valid={result.valid}, shape={result.features.shape}")
    print(f"   open_time: {result.open_time}")

    assert result.valid
    assert not np.any(np.isnan(result.features))

    for name, val in zip(CRYPTO_FEATURE_COLUMNS, result.features[0]):
        print(f"   {name:22s} = {val:+.6f}")

    print("\n✅ E2E тест пройден!")

asyncio.run(run())
