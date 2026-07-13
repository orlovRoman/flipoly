import asyncio
import sys
import os
import structlog

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polyflip.db.connection import async_session
from polyflip.models.trainer import ModelTrainer

logger = structlog.get_logger(__name__)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ml_pipeline.py <ASSET>")
        sys.exit(1)
        
    asset = sys.argv[1].upper()
    print(f"Запуск ML Pipeline для актива: {asset}")
    
    async with async_session() as session:
        trainer = ModelTrainer(session)
        success = await trainer.train_model(asset)
        
        if success:
            print(f"Успешно завершено обучение для {asset}")
        else:
            print(f"Обучение для {asset} завершилось с ошибкой или пропущено")

if __name__ == "__main__":
    asyncio.run(main())
