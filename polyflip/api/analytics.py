from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, Integer, update
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
import time
import asyncio

_probabilities_cache = None
_probabilities_cache_time = 0.0
_probabilities_lock = asyncio.Lock()

_summary_cache = None
_summary_cache_time = 0.0
_summary_lock = asyncio.Lock()

def invalidate_analytics_cache():
    global _summary_cache, _probabilities_cache
    _summary_cache = None
    _probabilities_cache = None


@router.get("/analytics/summary")
async def get_summary(db: AsyncSession = Depends(get_db_session)):
    """Общая статистика для дашборда"""
    global _summary_cache, _summary_cache_time
    now = time.time()
    if _summary_cache is not None and (now - _summary_cache_time) < 60:
        return _summary_cache

    async with _summary_lock:
        now = time.time()
        if _summary_cache is not None and (now - _summary_cache_time) < 60:
            return _summary_cache

    # 1. Считаем количество рынков и флипов
    total_markets_stmt = select(func.count(MarketSnapshot.id)).where(MarketSnapshot.final_outcome != "PENDING")
    total_markets = (await db.execute(total_markets_stmt)).scalar() or 0

    flips_stmt = select(func.count(MarketSnapshot.id)).where(
        MarketSnapshot.flip_vs_final,
        MarketSnapshot.final_outcome != "PENDING"
    )
    total_flips = (await db.execute(flips_stmt)).scalar() or 0

    # 2. Получаем текущие активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active)
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

    out = {
        "total_resolved_markets": total_markets,
        "total_flips": total_flips,
        "flip_percentage": round((total_flips / total_markets * 100) if total_markets > 0 else 0, 2),
        "active_models": active_models,
        "model_history": model_history
    }
    _summary_cache = out
    _summary_cache_time = now
    return out

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
    invalidate_analytics_cache()
    return {"status": "success", "active_version": version}

async def set_training_status(session: AsyncSession, asset: str, status: str, message: str, last_run: str = None):
    """Сохраняет статус обучения для конкретного актива в RuntimeSettings в виде JSON-строки."""
    key = f"TRAINING_STATUS_{asset.upper()}"
    logger.info("updating_training_status", asset=asset, status=status, message=message, last_run=last_run)
    if last_run is None:
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == key)
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
        key=key, 
        value=json.dumps(data),
        updated_at=datetime.now(timezone.utc),
        updated_by="train_job"
    )
    await session.merge(setting)
    await session.commit()

async def get_training_status(session: AsyncSession, asset: str) -> Dict[str, Any]:
    """Загружает статус обучения конкретного актива из RuntimeSettings."""
    key = f"TRAINING_STATUS_{asset.upper()}"
    stmt = select(RuntimeSettings).where(RuntimeSettings.key == key)
    res = await session.execute(stmt)
    row = res.scalar_one_or_none()
    if row:
        try:
            return json.loads(row.value)
        except Exception:
            pass
    return {"status": "idle", "message": "", "last_run": None}

@router.post("/analytics/train/{asset}", dependencies=[Depends(verify_api_key)])
async def trigger_training(asset: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db_session)):
    """Ручной запуск обучения моделей для конкретного актива"""
    asset = asset.upper()
    if asset not in settings.asset_list:
        raise HTTPException(status_code=400, detail=f"Актив {asset} не настроен в системе")
    
    now_iso = datetime.now(timezone.utc).isoformat()
    await set_training_status(db, asset, "running", f"Обучение для {asset} началось...", now_iso)
    
    # Чтобы не блокировать API-запрос, обучаем асинхронно
    async def train_single_asset():
        logger.info("train_single_asset_started", asset=asset)
        try:
            async with async_session() as bg_session:
                trainer = ModelTrainer(bg_session)
                try:
                    await trainer.train_model(asset)
                    msg = trainer.status_messages.get(asset, "Статус неизвестен")
                except Exception as e:
                    logger.exception("train_model_failed_for_asset", asset=asset, error=str(e))
                    msg = f"Ошибка: {str(e)}"
                
                logger.info("train_single_asset_completed", asset=asset, status=msg)
                await set_training_status(bg_session, asset, "success", f"{asset}: {msg}", datetime.now(timezone.utc).isoformat())
                invalidate_analytics_cache()
        except Exception as e:
            logger.exception("train_single_asset_failed", asset=asset, error=str(e))
            async with async_session() as bg_session:
                await set_training_status(bg_session, asset, "error", f"Ошибка: {str(e)}", datetime.now(timezone.utc).isoformat())
                invalidate_analytics_cache()
            
    background_tasks.add_task(train_single_asset)
    return {"status": "running", "asset": asset}

@router.get("/analytics/train_status/{asset}")
async def get_train_status(asset: str, db: AsyncSession = Depends(get_db_session)):
    """Возвращает статус последнего запущенного обучения для конкретного актива"""
    return await get_training_status(db, asset)

@router.get("/analytics/train_status")
async def get_train_status_all(db: AsyncSession = Depends(get_db_session)):
    """Возвращает статус обучения для всех активов"""
    keys = [f"TRAINING_STATUS_{a.upper()}" for a in settings.asset_list]
    stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(keys))
    rows = (await db.execute(stmt)).scalars().all()
    
    results = {a: {"status": "idle", "message": "", "last_run": None} for a in settings.asset_list}
    for row in rows:
        asset = row.key.replace("TRAINING_STATUS_", "")
        try:
            results[asset] = json.loads(row.value)
        except Exception:
            pass
    return results

@router.get("/analytics/probabilities")
async def get_flip_probabilities(db: AsyncSession = Depends(get_db_session)):
    """
    Возвращает вероятности изменения цены (флипа) в зависимости от всех параметров.
    """
    global _probabilities_cache, _probabilities_cache_time
    now = time.time()
    if _probabilities_cache is not None and (now - _probabilities_cache_time) < 300:
        return _probabilities_cache

    async with _probabilities_lock:
        now = time.time()
        if _probabilities_cache is not None and (now - _probabilities_cache_time) < 300:
            return _probabilities_cache

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
            
        # Материализуем строки в список словарей в основном потоке для потокобезопасности
        rows_data = [dict(r._mapping) for r in rows]
            
        def process_data(data):
            df = pd.DataFrame(data)
            
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
            
            out_data = {}
            
            for asset in df["asset"].unique():
                out_data[asset] = {}
                df_asset = df[df["asset"] == asset]
                
                for feature, (b, labels, right) in bins_config.items():
                    try:
                        binned = pd.cut(df_asset[feature], bins=b, labels=labels, right=right)
                        grouped = df_asset.groupby(binned, observed=False)["flip"].agg(['mean', 'count']).fillna(0)
                        
                        out_data[asset][feature] = {
                            "labels": labels,
                            "probabilities": [round(grouped.loc[lbl, 'mean'], 3) if lbl in grouped.index else 0 for lbl in labels],
                            "counts": [int(grouped.loc[lbl, 'count']) if lbl in grouped.index else 0 for lbl in labels]
                        }
                    except Exception as e:
                        structlog.get_logger(__name__).error("binning_error", feature=feature, error=str(e))
                        out_data[asset][feature] = {"labels": [], "probabilities": [], "counts": []}
            return out_data

        out = await asyncio.to_thread(process_data, rows_data)
        
        _probabilities_cache = out
        _probabilities_cache_time = now
        return out

# --- End Analytics ---
