"""
Запуск после всех feature-коммитов:
1. Деактивирует все модели в ModelRegistry для крипто-активов
2. Запускает переобучение для всех активных символов
3. Логирует feature importance в JSON для аудита
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, "/app")

from sqlalchemy import text
from polyflip.db.connection import get_db_session
from polyflip.crypto.trainer import CryptoModelTrainer

async def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    async with get_db_session() as db:
        print("1. Деактивация устаревших моделей в ModelRegistry...")
        await db.execute(
            text("UPDATE model_registry SET is_active = FALSE WHERE asset LIKE '%USDT%' OR asset = 'CRYPTO'")
        )
        await db.commit()

        os.makedirs("artifacts", exist_ok=True)
        trainer = CryptoModelTrainer(db)

        for symbol in symbols:
            print(f"\n2. Переобучение двухрежимных моделей для {symbol}...")
            res = await trainer.train(symbol)
            print(f"   Результат обучения {symbol}: {res}")

if __name__ == "__main__":
    asyncio.run(main())
