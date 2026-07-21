import asyncio
import os
from datetime import datetime, timezone
from sqlalchemy import select, func
from polyflip.db.connection import async_session
from polyflip.crypto.historical_loader import load_history_all
from polyflip.db.models import CryptoCandle, ModelRegistry, RuntimeSettings
from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.crypto.candle_repository import get_recent_candles

async def manual_backfill():
    print("\n--- [Step 5] Starting manual backfill ---")
    async with async_session() as session:
        results = await load_history_all(
            session,
            symbols=["DOGEUSDT", "XRPUSDT", "SOLUSDT"],
            intervals=["5m", "15m"],
            days_back=90,
        )
        print("Backfill results:", results)
        
        # Verify in DB
        stmt = select(
            CryptoCandle.symbol,
            CryptoCandle.interval,
            func.count().label("candles"),
            func.max(CryptoCandle.open_time).label("latest")
        ).where(
            CryptoCandle.symbol.in_(["DOGEUSDT", "XRPUSDT", "SOLUSDT"])
        ).group_by(
            CryptoCandle.symbol, CryptoCandle.interval
        ).order_by(
            CryptoCandle.symbol, CryptoCandle.interval
        )
        res = await session.execute(stmt)
        for row in res.all():
            print(f"DB Check: {row.symbol} {row.interval} - {row.candles} candles, latest: {row.latest}")
            
async def manual_train():
    print("\n--- [Step 6] Starting manual training ---")
    async with async_session() as session:
        trainer = CryptoModelTrainer(session)
        for symbol in ["DOGEUSDT", "XRPUSDT", "SOLUSDT"]:
            ok = await trainer.train(symbol, interval="15m")
            print(f"Train {symbol}: {'OK' if ok else 'FAILED'}")
            
        # Verify in DB
        stmt = select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.is_active,
            ModelRegistry.trained_at
        ).where(
            (ModelRegistry.asset.like("DOGE%")) | 
            (ModelRegistry.asset.like("XRP%")) | 
            (ModelRegistry.asset.like("SOL%"))
        ).order_by(
            ModelRegistry.asset, ModelRegistry.version.desc()
        )
        res = await session.execute(stmt)
        print("ModelRegistry entries:")
        for row in res.all():
            print(f"  {row.asset} v{row.version} active={row.is_active} trained={row.trained_at}")
            
        stmt2 = select(RuntimeSettings.key, RuntimeSettings.value).where(
            (RuntimeSettings.key.like("CRYPTO_THRESHOLD_DOGE%")) |
            (RuntimeSettings.key.like("CRYPTO_VOL_P%_DOGE%"))
        ).order_by(RuntimeSettings.key)
        res2 = await session.execute(stmt2)
        print("RuntimeSettings for DOGE:")
        for row in res2.all():
            print(f"  {row.key} = {row.value}")
            
async def manual_inference():
    print("\n--- [Step 7] Testing manual inference ---")
    async with async_session() as session:
        predictor = CryptoPredictor()
        ok = await predictor.load(session, "DOGEUSDT")
        assert ok, "Model for DOGEUSDT failed to load!"

        candles = await get_recent_candles(session, "DOGEUSDT", "15m", limit=120)
        signal = predictor.predict(candles, "DOGEUSDT")

        print(f"Prediction for DOGEUSDT: p_up={signal.p_up:.3f}, direction={signal.direction}, "
              f"version={signal.model_version}, features_ok={signal.features_ok}")
        
        assert signal.features_ok is True
        assert signal.model_version > 0
        assert signal.direction in ("UP", "DOWN", "NONE")

async def main():
    os.environ["PYTHONPATH"] = "."
    await manual_backfill()
    await manual_train()
    await manual_inference()

if __name__ == "__main__":
    asyncio.run(main())
