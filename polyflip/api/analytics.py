from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, Integer, update
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import structlog
import json
from datetime import datetime, timezone

from polyflip.db.connection import get_db_session, async_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.config import settings
from polyflip.models.trainer import ModelTrainer

logger = structlog.get_logger(__name__)

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
    
    # 3. История моделей для графиков
    history_stmt = select(ModelRegistry).order_by(ModelRegistry.trained_at.asc())
    history_rows = (await db.execute(history_stmt)).scalars().all()
    
    model_history = {}
    for r in history_rows:
        if r.asset not in model_history:
            model_history[r.asset] = []
        model_history[r.asset].append({
            "version": r.version,
            "accuracy": round(r.accuracy, 4),
            "trained_at": r.trained_at.isoformat() if r.trained_at else None
        })

    return {
        "total_resolved_markets": total_markets,
        "total_flips": total_flips,
        "flip_percentage": round((total_flips / total_markets * 100) if total_markets > 0 else 0, 2),
        "active_models": active_models,
        "model_history": model_history
    }

@router.get("/analytics/models")
async def list_models(db: AsyncSession = Depends(get_db_session)):
    """Получение истории всех моделей"""
    stmt = select(ModelRegistry).order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    models = (await db.execute(stmt)).scalars().all()
    return [
        {
            "asset": m.asset,
            "version": m.version,
            "accuracy": round(m.accuracy, 4),
            "baseline": round(m.baseline, 4) if m.baseline is not None else None,
            "features": m.features or "",
            "is_active": m.is_active,
            "trained_at": m.trained_at.isoformat() if m.trained_at else None
        }
        for m in models
    ]

@router.post("/analytics/models/{asset}/activate/{version}", dependencies=[Depends(verify_api_key)])
async def activate_model(asset: str, version: int, db: AsyncSession = Depends(get_db_session)):
    """Смена активной модели"""
    # Деактивируем все модели этого актива
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset)
        .values(is_active=False)
    )
    
    # Активируем выбранную
    result = await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset, ModelRegistry.version == version)
        .values(is_active=True)
    )
    
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Model not found")
        
    await db.commit()
    return {"status": "success", "active_version": version}

async def set_training_status(session: AsyncSession, status: str, message: str, last_run: str = None):
    """Сохраняет статус обучения в RuntimeSettings в виде JSON-строки."""
    logger.info("updating_training_status", status=status, message=message, last_run=last_run)
    if last_run is None:
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRAINING_STATUS_JSON")
        res = await session.execute(stmt)
        row = res.scalar_one_or_none()
        if row:
            try:
                old_data = json.loads(row.value)
                last_run = old_data.get("last_run")
            except Exception:
                pass
    
    data = {
        "status": status,
        "message": message,
        "last_run": last_run
    }
    setting = RuntimeSettings(
        key="TRAINING_STATUS_JSON", 
        value=json.dumps(data),
        updated_at=datetime.now(timezone.utc),
        updated_by="train_job"
    )
    await session.merge(setting)
    await session.commit()

async def get_training_status(session: AsyncSession) -> Dict[str, Any]:
    """Загружает статус обучения из RuntimeSettings."""
    stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRAINING_STATUS_JSON")
    res = await session.execute(stmt)
    row = res.scalar_one_or_none()
    if row:
        try:
            return json.loads(row.value)
        except Exception:
            pass
    return {"status": "idle", "message": "", "last_run": None}

@router.post("/analytics/train", dependencies=[Depends(verify_api_key)])
async def trigger_training(background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db_session)):
    """Ручной запуск обучения моделей по всем активам"""
    
    now_iso = datetime.now(timezone.utc).isoformat()
    await set_training_status(db, "running", "Обучение началось...", now_iso)
    
    # Чтобы не блокировать API-запрос, обучаем асинхронно
    async def train_all():
        logger.info("train_all_started")
        try:
            async with async_session() as bg_session:
                trainer = ModelTrainer(bg_session)
                for asset in settings.asset_list:
                    try:
                        await trainer.train_model(asset)
                    except Exception as e:
                        logger.exception("train_model_failed_for_asset", asset=asset, error=str(e))
                        trainer.status_messages[asset] = f"Ошибка: {str(e)}"
                
                # Собираем отчеты по обучению
                summary_msgs = []
                for asset in settings.asset_list:
                    msg = trainer.status_messages.get(asset, "Статус неизвестен")
                    summary_msgs.append(f"{asset}: {msg}")
                summary_message = " | ".join(summary_msgs)
                
                logger.info("train_all_completed", summary=summary_message)
                await set_training_status(bg_session, "success", summary_message, datetime.now(timezone.utc).isoformat())
        except Exception as e:
            logger.exception("train_all_failed", error=str(e))
            async with async_session() as bg_session:
                await set_training_status(bg_session, "error", f"Ошибка: {str(e)}", datetime.now(timezone.utc).isoformat())
            
    background_tasks.add_task(train_all)
    return {"status": "training_started"}

@router.get("/analytics/train_status")
async def get_train_status(db: AsyncSession = Depends(get_db_session)):
    """Возвращает статус последнего запущенного обучения"""
    return await get_training_status(db)

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
    ).where(MarketSnapshot.final_outcome != "PENDING").order_by(MarketSnapshot.recorded_at.desc()).limit(150000)
    
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
                structlog.get_logger(__name__).error("binning_error", feature=feature, error=str(e))
                out[asset][feature] = {"labels": [], "probabilities": [], "counts": []}
                
    return out

# --- End Analytics ---
