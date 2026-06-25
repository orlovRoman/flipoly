import pickle
import pandas as pd
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from polyflip.db.models import MarketSnapshot, ModelRegistry

logger = structlog.get_logger(__name__)

class ModelTrainer:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def train_model(self, asset: str) -> bool:
        """
        Обучает модель LogisticRegression для заданного актива на основе 
        исторических (разрезолвленных) данных и сохраняет в БД.
        """
        logger.info("starting_training", asset=asset)
        
        # 1. Получаем обучающую выборку (исключаем PENDING)
        stmt = select(MarketSnapshot).where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.final_outcome != "PENDING"
        )
        result = await self.db.execute(stmt)
        snapshots = result.scalars().all()

        if not snapshots:
            logger.warning("no_training_data_found", asset=asset)
            return False

        # 2. Формируем DataFrame
        data = []
        for s in snapshots:
            data.append({
                "time_left_min": s.time_left_min,
                "mid_price": s.mid_price,
                "spread": s.spread,
                "price_velocity": s.price_velocity,
                "volume_5min": s.volume_5min,
                "target": 1 if s.flip_vs_final else 0
            })
            
        df = pd.DataFrame(data)
        
        # Базовая проверка на разнообразие классов
        if len(df["target"].unique()) < 2:
            logger.warning("only_one_class_in_target", asset=asset)
            return False
            
        X = df[["time_left_min", "mid_price", "spread", "price_velocity", "volume_5min"]]
        y = df["target"]

        # 3. Обучаем модель (пока без гиперпараметрического поиска для простоты)
        model = LogisticRegression(class_weight="balanced", random_state=42)
        model.fit(X, y)
        
        # Вычисляем accuracy на тренировочной выборке (просто для логов)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        logger.info("model_trained", asset=asset, samples=len(df), accuracy=acc)

        # 4. Сериализуем модель
        model_bytes = pickle.dumps(model)

        # 5. Деактивируем предыдущие модели
        await self.db.execute(
            update(ModelRegistry)
            .where(ModelRegistry.asset == asset)
            .values(is_active=False)
        )

        # 6. Получаем следующий номер версии
        version_stmt = select(ModelRegistry.version).where(ModelRegistry.asset == asset).order_by(ModelRegistry.version.desc()).limit(1)
        v_result = await self.db.execute(version_stmt)
        last_v = v_result.scalar_one_or_none()
        next_version = (last_v or 0) + 1

        # 7. Сохраняем новую модель
        new_model_record = ModelRegistry(
            asset=asset,
            version=next_version,
            model_blob=model_bytes,
            accuracy=acc,
            is_active=True,
            trained_at=datetime.now(timezone.utc)
        )
        self.db.add(new_model_record)
        await self.db.commit()

        logger.info("model_saved_to_db", asset=asset, version=next_version)
        return True
