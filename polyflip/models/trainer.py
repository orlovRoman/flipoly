import pickle
import pandas as pd
import asyncio
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.config import settings

logger = structlog.get_logger(__name__)

def _fit_and_serialize(X: pd.DataFrame, y: pd.Series):
    """Синхронная CPU-bound функция для кросс-валидации, обучения и сериализации модели."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import pickle

    # 3. Обучаем модель с кросс-валидацией
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = LogisticRegression(class_weight="balanced", random_state=42)
    
    accuracies = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model_clone = clone(base_model)
        model_clone.fit(X_train, y_train)
        
        preds = model_clone.predict(X_val)
        accuracies.append(accuracy_score(y_val, preds))
        
    val_acc = float(np.mean(accuracies))
    
    # Baseline
    baseline_acc = float(max(y.mean(), 1 - y.mean()))
    
    # Обучаем финальную модель на всех данных (ограничение убрано)
    final_model = LogisticRegression(class_weight="balanced", random_state=42)
    final_model.fit(X, y)
    
    # Сериализуем модель
    model_bytes = pickle.dumps(final_model)
    return model_bytes, val_acc, baseline_acc

class ModelTrainer:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.status_messages = {}

    async def train_model(self, asset: str) -> bool:
        """
        Обучает модель LogisticRegression для заданного актива на основе 
        исторических (разрезолвленных) данных и сохраняет в БД.
        """
        logger.info("starting_training", asset=asset)
        
        # Получаем активные фичи из RuntimeSettings
        settings_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "ACTIVE_FEATURES")
        settings_result = await self.db.execute(settings_stmt)
        active_features_setting = settings_result.scalar_one_or_none()
        
        if active_features_setting and active_features_setting.value.strip():
            active_features = active_features_setting.value.split(",")
        else:
            active_features = settings.ACTIVE_FEATURES.split(",")
            
        active_features = [f.strip() for f in active_features if f.strip()]
        
        if not active_features:
            logger.error("no_active_features_selected", asset=asset)
            self.status_messages[asset] = "Ошибка: не выбраны активные признаки"
            return False
        
        # 1. Получаем обучающую выборку (исключаем PENDING)
        stmt = select(MarketSnapshot).where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.final_outcome != "PENDING"
        )
        result = await self.db.execute(stmt)
        snapshots = result.scalars().all()

        # BUG-004 FIX: Используем настройку из конфига
        if len(snapshots) < settings.MIN_SAMPLES_FOR_MODEL:
            logger.warning("not_enough_data_for_training", asset=asset, samples=len(snapshots), required=settings.MIN_SAMPLES_FOR_MODEL)
            self.status_messages[asset] = f"Пропущено: недостаточно данных ({len(snapshots)}/{settings.MIN_SAMPLES_FOR_MODEL})"
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
                "hour_of_day": s.hour_of_day, # BUG-006 FIX: Добавлена фича hour_of_day
                "target": 1 if s.flip_vs_final else 0
            })
            
        df = pd.DataFrame(data)
        
        # Базовая проверка на разнообразие классов
        if len(df["target"].unique()) < 2:
            logger.warning("only_one_class_in_target", asset=asset)
            self.status_messages[asset] = "Пропущено: все исходы одинаковы (1 класс)"
            return False
            
        # Используем только те фичи, которые включены в дашборде
        missing_features = [f for f in active_features if f not in df.columns]
        if missing_features:
            logger.error("missing_features_in_df", missing=missing_features)
            self.status_messages[asset] = f"Ошибка: отсутствуют фичи {', '.join(missing_features)}"
            return False
            
        X = df[active_features]
        y = df["target"]

        # Выполняем CPU-bound обучение в отдельном потоке (BUG-A2 FIX)
        fit_res = await asyncio.to_thread(_fit_and_serialize, X, y)
        model_bytes, val_acc, baseline_acc = fit_res

        logger.info("model_trained", asset=asset, samples=len(df), val_accuracy=val_acc, baseline_accuracy=baseline_acc)

        # Деактивируем предыдущие модели
        await self.db.execute(
            update(ModelRegistry)
            .where(ModelRegistry.asset == asset)
            .values(is_active=False)
        )

        # Получаем следующий номер версии
        version_stmt = select(ModelRegistry.version).where(ModelRegistry.asset == asset).order_by(ModelRegistry.version.desc()).limit(1)
        v_result = await self.db.execute(version_stmt)
        last_v = v_result.scalar_one_or_none()
        next_version = (last_v or 0) + 1

        # 7. Сохраняем новую модель
        new_model_record = ModelRegistry(
            asset=asset,
            version=next_version,
            model_blob=model_bytes,
            accuracy=val_acc,
            baseline=baseline_acc,
            features=",".join(active_features),
            is_active=True,
            trained_at=datetime.now(timezone.utc)
        )
        self.db.add(new_model_record)
        await self.db.commit()

        logger.info("model_saved_to_db", asset=asset, version=next_version)
        self.status_messages[asset] = f"Успешно: версия {next_version} (точность {val_acc:.2%}, бейзлайн {baseline_acc:.2%})"
        return True
