from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, Integer
from pydantic import BaseModel
from typing import Dict, Any

from polyflip.db.connection import get_db_session, async_session
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

    flips_stmt = select(func.count(MarketSnapshot.id)).where(
        MarketSnapshot.flip_vs_final == True,
        MarketSnapshot.final_outcome != "PENDING"
    )
    total_flips = (await db.execute(flips_stmt)).scalar() or 0

    # 2. Получаем текущие активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active == True)
    models = (await db.execute(models_stmt)).scalars().all()
    
    active_models = {
        m.asset: {
            "version": m.version,
            "accuracy": round(m.accuracy, 4),
            "trained_at": m.trained_at
        }
        for m in models
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
    
    # Чтобы не блокировать API-запрос, обучаем асинхронно
    async def train_all():
        async with async_session() as bg_session:
            trainer = ModelTrainer(bg_session)
            for asset in settings.asset_list:
                await trainer.train_model(asset)
            
    background_tasks.add_task(train_all)
    return {"status": "training_started"}

import pandas as pd

@router.get("/analytics/probabilities")
async def get_flip_probabilities(db: AsyncSession = Depends(get_db_session)):
    """
    Возвращает вероятности изменения цены (флипа) в зависимости от всех параметров.
    """
    stmt = select(
        MarketSnapshot.asset,
        cast(MarketSnapshot.flip_vs_final, Integer).label("flip"),
        MarketSnapshot.time_left_min,
        MarketSnapshot.mid_price,
        MarketSnapshot.spread,
        MarketSnapshot.volume_5min,
        MarketSnapshot.price_velocity,
        MarketSnapshot.hour_of_day
    ).where(MarketSnapshot.final_outcome != "PENDING")
    
    result = await db.execute(stmt)
    rows = result.all()
    
    if not rows:
        return {}
        
    df = pd.DataFrame([dict(r._mapping) for r in rows])
    
    # Define bins: (edges, labels, right_closed)
    bins_config = {
        "time_left_min": (list(range(17)), [str(i) for i in range(16)], False),
        "mid_price": ([-0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], 
                      ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"], True),
        "spread": ([-0.01, 0.01, 0.02, 0.03, 0.05, 0.1, 100.0], 
                   ["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.05", "0.05-0.10", ">0.10"], True),
        "volume_5min": ([-1, 100, 1000, 5000, 10000, 50000, 1e9], 
                        ["0-100", "100-1k", "1k-5k", "5k-10k", "10k-50k", ">50k"], True),
        "price_velocity": ([-100, -0.05, -0.01, -0.001, 0.001, 0.01, 0.05, 100], 
                           ["<-5%", "-5% to -1%", "-1% to 0%", "0", "0% to 1%", "1% to 5%", ">5%"], True),
        "hour_of_day": (list(range(25)), [str(i) for i in range(24)], False)
    }
    
    out = {}
    
    for asset in df["asset"].unique():
        out[asset] = {}
        df_asset = df[df["asset"] == asset]
        
        for feature, (b, labels, right) in bins_config.items():
            try:
                binned = pd.cut(df_asset[feature], bins=b, labels=labels, right=right)
                grouped = df_asset.groupby(binned, observed=False)["flip"].agg(['mean', 'count']).fillna(0)
                
                out[asset][feature] = {
                    "labels": labels,
                    "probabilities": [round(grouped.loc[lbl, 'mean'], 3) if lbl in grouped.index else 0 for lbl in labels],
                    "counts": [int(grouped.loc[lbl, 'count']) if lbl in grouped.index else 0 for lbl in labels]
                }
            except Exception as e:
                import structlog
                structlog.get_logger(__name__).error("binning_error", feature=feature, error=str(e))
                out[asset][feature] = {"labels": [], "probabilities": [], "counts": []}
                
    return out

# --- End Analytics ---
