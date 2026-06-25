from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Dict, Any

from polyflip.db.connection import get_db_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.config import settings
from polyflip.models.trainer import ModelTrainer

router = APIRouter(prefix="/api", tags=["Analytics & Settings"])

# --- Analytics ---

@router.get("/analytics/summary")
async def get_summary(db: AsyncSession = Depends(get_db_session)):
    """Общая статистика для дашборда"""
    # 1. Считаем количество рынков и флипов
    total_markets_stmt = select(func.count(MarketSnapshot.id)).where(MarketSnapshot.final_outcome != "PENDING")
    total_markets = (await db.execute(total_markets_stmt)).scalar() or 0

    flips_stmt = select(func.count(MarketSnapshot.id)).where(MarketSnapshot.flip_vs_final == True)
    total_flips = (await db.execute(flips_stmt)).scalar() or 0

    # 2. Получаем текущие активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active == True)
    models = (await db.execute(models_stmt)).scalars().all()
    
    active_models = {}
    for m in models:
        active_models[m.asset] = {
            "version": m.version,
            "accuracy": round(m.accuracy, 4),
            "trained_at": m.trained_at
        }

    return {
        "total_resolved_markets": total_markets,
        "total_flips": total_flips,
        "flip_percentage": round((total_flips / total_markets * 100) if total_markets > 0 else 0, 2),
        "active_models": active_models
    }

@router.post("/analytics/train", dependencies=[Depends(verify_api_key)])
async def trigger_training(background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db_session)):
    """Ручной запуск обучения моделей по всем активам"""
    trainer = ModelTrainer(db)
    
    # Чтобы не блокировать API-запрос, обучаем асинхронно или быстро
    # Для текущей архитектуры (logistic regression на небольшом датасете) можно в синхронном режиме, 
    # но лучше обернуть
    async def train_all():
        for asset in settings.asset_list:
            await trainer.train_model(asset)
            
    background_tasks.add_task(train_all)
    return {"status": "training_started"}

@router.get("/analytics/probabilities")
async def get_flip_probabilities(db: AsyncSession = Depends(get_db_session)):
    """
    Возвращает вероятности изменения цены (флипа) в зависимости от оставшегося времени (0-15 минут)
    для графиков по ETH и BTC.
    """
    # Группируем по asset и округленному времени (time_left_min)
    # Используем CAST(time_left_min AS INTEGER) для агрегации по минутам
    stmt = select(
        MarketSnapshot.asset,
        func.cast(MarketSnapshot.time_left_min, func.Integer).label("minute"),
        func.count(MarketSnapshot.id).label("total"),
        func.sum(func.cast(MarketSnapshot.flip_vs_final, func.Integer)).label("flips")
    ).where(
        MarketSnapshot.final_outcome != "PENDING",
        MarketSnapshot.time_left_min <= 16.0,
        MarketSnapshot.time_left_min >= 0.0
    ).group_by(
        MarketSnapshot.asset,
        func.cast(MarketSnapshot.time_left_min, func.Integer)
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    # Форматируем данные: { "BTC": { "15": 0.1, "14": 0.15, ... }, "ETH": { ... } }
    data = {}
    for row in rows:
        asset = row.asset
        minute = str(row.minute)
        prob = row.flips / row.total if row.total > 0 else 0
        
        if asset not in data:
            data[asset] = {}
        data[asset][minute] = round(prob, 3)
        
    return data

# --- Settings ---

class SettingUpdate(BaseModel):
    value: str
    updated_by: str = "api"

@router.get("/settings")
async def get_all_settings(db: AsyncSession = Depends(get_db_session)):
    """Возвращает все рантайм-настройки"""
    result = await db.execute(select(RuntimeSettings))
    db_settings = result.scalars().all()
    
    return {s.key: s.value for s in db_settings}

@router.put("/settings/{key}", dependencies=[Depends(verify_api_key)])
async def update_setting(key: str, payload: SettingUpdate, db: AsyncSession = Depends(get_db_session)):
    """Обновляет значение настройки"""
    from datetime import datetime, timezone
    
    result = await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = payload.value
        setting.updated_at = datetime.now(timezone.utc)
        setting.updated_by = payload.updated_by
    else:
        setting = RuntimeSettings(
            key=key,
            value=payload.value,
            updated_at=datetime.now(timezone.utc),
            updated_by=payload.updated_by
        )
        db.add(setting)
        
    await db.commit()
    return {"status": "success", "key": key, "value": payload.value}
