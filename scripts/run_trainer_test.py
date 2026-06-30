import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
import random

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polyflip.db.models import MarketSnapshot, ModelRegistry
from polyflip.db.connection import async_session
from polyflip.models.trainer import ModelTrainer
from sqlalchemy import select, delete

async def main():
    print("Генерация фейковых данных для теста ModelTrainer...")
    
    async with async_session() as session:
        # Очищаем старые тестовые данные
        await session.execute(delete(MarketSnapshot).where(MarketSnapshot.asset == "TEST_BTC"))
        await session.execute(delete(ModelRegistry).where(ModelRegistry.asset == "TEST_BTC"))
        await session.commit()
        
        # Создаем 100 фейковых рынков
        now = datetime.now(timezone.utc)
        for i in range(100):
            # Имитируем логику: если time_left_min < 5 и spread узкий, flip менее вероятен
            # Это просто рандом с небольшим смещением для логистической регрессии
            time_left = random.uniform(1.0, 15.0)
            mid_price = random.uniform(0.1, 0.9)
            spread = random.uniform(0.01, 0.1)
            velocity = random.uniform(-0.05, 0.05)
            vol = random.uniform(1000, 50000)
            
            # Немного логики для создания обучаемой закономерности
            is_flip = (mid_price > 0.7 and velocity < 0) or (mid_price < 0.3 and velocity > 0)
            
            snap = MarketSnapshot(
                asset="TEST_BTC",
                market_id=f"test_market_{i}",
                time_left_min=time_left,
                mid_price=mid_price,
                spread=spread,
                volume_5min=vol,
                price_velocity=velocity,
                hour_of_day=now.hour,
                final_outcome="YES" if random.random() > 0.5 else "NO",
                flip_vs_final=is_flip,
                recorded_at=now - timedelta(days=1)
            )
            session.add(snap)
            
        await session.commit()
        print("Данные сгенерированы. Запускаем тренинг...")
        
        trainer = ModelTrainer(session)
        success = await trainer.train_model("TEST_BTC")
        
        if success:
            print("Тренинг успешно завершен!")
            # Проверим, что модель в базе
            result = await session.execute(select(ModelRegistry).where(ModelRegistry.asset == "TEST_BTC"))
            model_record = result.scalars().first()
            if model_record:
                print(f"Модель найдена в БД! Версия: {model_record.version}, Точность: {model_record.accuracy:.2f}, Размер blob: {len(model_record.model_blob)} байт")
            else:
                print("ОШИБКА: Модель не найдена в БД после успешного тренинга.")
        else:
            print("ОШИБКА: Тренинг вернул False.")

if __name__ == "__main__":
    asyncio.run(main())
