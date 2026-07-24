from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, Integer, update, delete
from typing import Dict, Any
import pickle
import pandas as pd
import structlog
import json
from datetime import datetime, timezone, timedelta

from polyflip.db.connection import get_db_session, async_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings, TradeHistory

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

_time_left_dist_cache = None
_time_left_dist_cache_time = 0.0
_time_left_dist_lock = asyncio.Lock()

async def invalidate_analytics_cache():
    global _summary_cache, _probabilities_cache, _time_left_dist_cache
    async with _summary_lock:
        _summary_cache = None
    async with _probabilities_lock:
        _probabilities_cache = None
    async with _time_left_dist_lock:
        _time_left_dist_cache = None


@router.get("/analytics/summary")
async def get_summary(db: AsyncSession = Depends(get_db_session)):
    """Общая статистика для дашборда"""
    global _summary_cache, _summary_cache_time
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
            "ece": round(getattr(m, 'ece', 0.0), 4) if getattr(m, 'ece', None) is not None else None,
            "trained_at": m.trained_at.isoformat() if m.trained_at else None
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
            "ece": round(getattr(r, 'ece', 0.0), 4) if getattr(r, 'ece', None) is not None else None,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None
        })

    out = {
        "total_resolved_markets": total_markets,
        "total_flips": total_flips,
        "flip_percentage": round((total_flips / total_markets * 100) if total_markets > 0 else 0, 2),
        "active_models": active_models,
        "model_history": model_history
    }
    
    async with _summary_lock:
        if _summary_cache is None or (time.time() - _summary_cache_time) >= 60:
            _summary_cache = out
            _summary_cache_time = time.time()
            
    return _summary_cache

def get_model_type(asset: str) -> tuple[str, str]:
    if any(asset.endswith(s) for s in ["_low_vol", "_mid_vol", "_high_vol"]):
        return ("lightgbm", "LightGBM (Crypto)")
    return ("logistic_regression", "Logistic Regression (Phase)")


def extract_coefficients_from_blob(model_blob: bytes, features_str: str) -> dict:
    """Извлекает весовые коэффициенты признаков из сериализованного sklearn объекта (Pipeline / CalibratedClassifierCV)."""
    if not model_blob or not features_str:
        return {}
    
    feature_names = [f.strip() for f in features_str.split(",") if f.strip()]
    if not feature_names:
        return {}

    try:
        model_obj = pickle.loads(model_blob)
        base_estimator = model_obj
        
        if hasattr(model_obj, "calibrated_classifiers_") and model_obj.calibrated_classifiers_:
            cc = model_obj.calibrated_classifiers_[0]
            base_estimator = getattr(cc, "estimator", cc)
            if hasattr(base_estimator, "estimator"):
                base_estimator = base_estimator.estimator

        if hasattr(base_estimator, "named_steps") and "model" in base_estimator.named_steps:
            lr_model = base_estimator.named_steps["model"]
            if hasattr(lr_model, "coef_"):
                coefs = lr_model.coef_[0]
                if len(coefs) == len(feature_names):
                    return {f: round(float(c), 4) for f, c in zip(feature_names, coefs)}
    except Exception as e:
        logger.warning("failed_to_extract_coefficients", error=str(e))

    return {}


from sqlalchemy.orm import defer
import time

_models_cache = {}
_MODELS_CACHE_TTL = 10.0

def invalidate_models_cache():
    _models_cache.clear()

@router.get("/analytics/models")
async def list_models(db: AsyncSession = Depends(get_db_session)):
    """Получение истории ВСЕХ моделей с мгновенной загрузкой метаданных без скачивания model_blob"""
    now = time.time()
    if "data" in _models_cache and now - _models_cache.get("time", 0) < _MODELS_CACHE_TTL:
        return _models_cache["data"]

    stmt = (
        select(ModelRegistry)
        .options(defer(ModelRegistry.model_blob))
        .order_by(ModelRegistry.trained_at.desc(), ModelRegistry.version.desc())
    )
    models = (await db.execute(stmt)).scalars().all()

    result = []
    for m in models:
        m_type, algo_label = get_model_type(m.asset)
        lift_val = round(m.accuracy - m.baseline, 4) if (m.accuracy is not None and m.baseline is not None) else None
        result.append({
            "asset": m.asset,
            "version": m.version,
            "accuracy": round(m.accuracy, 4) if m.accuracy is not None else None,
            "baseline": round(m.baseline, 4) if m.baseline is not None else None,
            "lift": lift_val,
            "ece": round(getattr(m, 'ece', 0.0), 4) if getattr(m, 'ece', None) is not None else None,
            "features": m.features or "",
            "is_active": m.is_active,
            "trained_at": m.trained_at.isoformat() if m.trained_at else None,
            "model_type": m_type,
            "algorithm": algo_label
        })

    _models_cache["time"] = now
    _models_cache["data"] = result
    return result


@router.get("/analytics/models/{asset}/{version}/coefficients")
async def get_model_coefficients(asset: str, version: int, db: AsyncSession = Depends(get_db_session)):
    """Ленивая загрузка весовых коэффициентов для выбранной модели"""
    stmt = (
        select(ModelRegistry.model_blob, ModelRegistry.features)
        .where(ModelRegistry.asset == asset, ModelRegistry.version == version)
        .limit(1)
    )
    row = (await db.execute(stmt)).first()
    if not row or not row.model_blob:
        return {}
    return extract_coefficients_from_blob(row.model_blob, row.features)


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
    
    await db.commit()
    invalidate_models_cache()
    await invalidate_analytics_cache()
    return {"status": "success", "active_version": version}


@router.delete("/analytics/models/{asset}/{version}", dependencies=[Depends(verify_api_key)])
async def delete_model(asset: str, version: int, db: AsyncSession = Depends(get_db_session)):
    """Удаление архивной модели"""
    check_stmt = select(ModelRegistry.is_active).where(
        ModelRegistry.asset == asset, ModelRegistry.version == version
    )
    is_active = (await db.execute(check_stmt)).scalar_one_or_none()
    if is_active is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if is_active:
        raise HTTPException(status_code=400, detail="Cannot delete active model. Activate another model first.")

    from sqlalchemy import delete
    del_stmt = delete(ModelRegistry).where(
        ModelRegistry.asset == asset, ModelRegistry.version == version
    )
    await db.execute(del_stmt)
    await db.commit()
    invalidate_models_cache()
    await invalidate_analytics_cache()
    return {"status": "success", "message": f"Model {asset} v{version} deleted"}


def get_model_subtype_info(asset: str) -> tuple[str, str, str]:
    """
    Возвращает (base_symbol, subtype_code, subtype_label)
    """
    base_symbol = asset.split('_')[0].replace('USDT', '').upper()
    if asset.endswith("_leaning"):
        return (base_symbol, "leaning", "🟨 Фаза колебаний (Leaning LogReg)")
    elif asset.endswith("_decided"):
        return (base_symbol, "decided", "🟦 Определившийся рынок (Decided LogReg)")
    elif asset.endswith("_contested"):
        return (base_symbol, "contested", "🔴 Острая борьба (Contested LogReg)")
    elif asset.endswith("_low_vol"):
        return (base_symbol, "lightgbm_low", "🔷 LightGBM (Low Vol)")
    elif asset.endswith("_mid_vol"):
        return (base_symbol, "lightgbm_mid", "🔷 LightGBM (Mid Vol)")
    elif asset.endswith("_high_vol"):
        return (base_symbol, "lightgbm_high", "🔷 LightGBM (High Vol)")
    else:
        return (base_symbol, "base", "🟧 Базовая модель (Base LogReg)")


@router.get("/analytics/active_models_summary", dependencies=[Depends(verify_api_key)])
async def get_active_models_summary(timeframe: str = "24h", db: AsyncSession = Depends(get_db_session)):
    """
    Сводная статистика по всем АКТИВНЫМ моделям (Base, Leaning, Decided, Contested, LightGBM)
    с расчетом PnL, WinRate и количества сделок за выбранный период.
    """
    now = datetime.now(timezone.utc)
    if timeframe == "24h":
        start_time = now - timedelta(hours=24)
    elif timeframe == "7d":
        start_time = now - timedelta(days=7)
    elif timeframe == "30d":
        start_time = now - timedelta(days=30)
    else:
        start_time = None

    # 1. Запрашиваем активные модели из ModelRegistry
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active).order_by(ModelRegistry.asset)
    models = (await db.execute(models_stmt)).scalars().all()

    # 2. Запрашиваем успешные сделки
    trades_stmt = select(TradeHistory).where(
        TradeHistory.status == "SUCCESS",
        TradeHistory.pnl.is_not(None),
        TradeHistory.model_version.is_not(None)
    )
    if start_time:
        trades_stmt = trades_stmt.where(TradeHistory.created_at >= start_time)

    trades = (await db.execute(trades_stmt)).scalars().all()

    LGBM_SUFFIXES = ("_low_vol", "_mid_vol", "_high_vol")

    # Группируем трейды двумя способами:
    # 1. trades_by_exact  — по полному имени актива (для LightGBM)
    # 2. trades_by_base   — по базовому символу (для LogReg: base/leaning/decided/contested)
    trades_by_exact: dict[tuple, dict] = {}
    trades_by_base:  dict[tuple, dict] = {}

    for t in trades:
        norm_asset = t.asset.split('_')[0].replace('USDT', '').upper()

        # Exact (LightGBM)
        k_exact = (t.asset, t.model_version)
        if k_exact not in trades_by_exact:
            trades_by_exact[k_exact] = {"total": 0, "wins": 0, "pnl": 0.0}
        trades_by_exact[k_exact]["total"] += 1
        if t.pnl > 0:
            trades_by_exact[k_exact]["wins"] += 1
        trades_by_exact[k_exact]["pnl"] += float(t.pnl)

        # Base (LogReg — все фазовые модели BTC/ETH/SOL пишут просто "BTC")
        k_base = (norm_asset, t.model_version)
        if k_base not in trades_by_base:
            trades_by_base[k_base] = {"total": 0, "wins": 0, "pnl": 0.0}
        trades_by_base[k_base]["total"] += 1
        if t.pnl > 0:
            trades_by_base[k_base]["wins"] += 1
        trades_by_base[k_base]["pnl"] += float(t.pnl)

    result = []
    for m in models:
        base_symbol, sub_code, sub_label = get_model_subtype_info(m.asset)

        is_lgbm = any(m.asset.endswith(s) for s in LGBM_SUFFIXES)

        if is_lgbm:
            # LightGBM: строгий exact-матч по полному имени (нет коллизий между low/mid/high_vol)
            stats = trades_by_exact.get((m.asset, m.version), {"total": 0, "wins": 0, "pnl": 0.0})
        else:
            # LogReg (base/leaning/decided/contested): матч по базовому символу
            # т.к. в TradeHistory.asset всегда записан чистый тикер "BTC", "ETH", "SOL"
            stats = trades_by_base.get((base_symbol, m.version), {"total": 0, "wins": 0, "pnl": 0.0})

        win_rate = round((stats["wins"] / stats["total"] * 100), 1) if stats["total"] > 0 else None

        result.append({
            "asset_full":    m.asset,
            "base_symbol":   base_symbol,
            "subtype_code":  sub_code,
            "subtype_label": sub_label,
            "version":       m.version,
            "accuracy":      round(m.accuracy, 4) if m.accuracy is not None else None,
            "ece":           round(getattr(m, 'ece', 0.0), 4) if getattr(m, 'ece', None) is not None else None,
            "total_trades":  stats["total"],
            "win_rate":      win_rate,
            "pnl":           round(stats["pnl"], 2),
            "trained_at":    m.trained_at.isoformat() if m.trained_at else None,
        })

    return {"status": "success", "timeframe": timeframe, "data": result}






@router.delete("/analytics/models/{asset}/{version}", dependencies=[Depends(verify_api_key)])
async def delete_model(asset: str, version: int, db: AsyncSession = Depends(get_db_session)):
    """Удаление архивной модели"""
    stmt = select(ModelRegistry).where(ModelRegistry.asset == asset, ModelRegistry.version == version)
    model = (await db.execute(stmt)).scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    if model.is_active:
        raise HTTPException(status_code=400, detail="Cannot delete active model")
        
    await db.execute(delete(ModelRegistry).where(ModelRegistry.asset == asset, ModelRegistry.version == version))
    await db.commit()
    await invalidate_analytics_cache()
    return {"status": "success", "deleted_version": version}

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
                await invalidate_analytics_cache()
        except Exception as e:
            logger.exception("train_single_asset_failed", asset=asset, error=str(e))
            async with async_session() as bg_session:
                await set_training_status(bg_session, asset, "error", f"Ошибка: {str(e)}", datetime.now(timezone.utc).isoformat())
                await invalidate_analytics_cache()
            
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
        time_left_edges  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,float("inf")]
        time_left_labels = ["0m","1m","2m","3m","4m","5m","6m","7m","8m","9m",
                            "10m","11m","12m","13m","14m","15m","16m","17m","18m","19m","20m","21m",">22m"]
        bins_config = {
            "time_left_min": (time_left_edges, time_left_labels, False),
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
    
    async with _probabilities_lock:
        if _probabilities_cache is None or (time.time() - _probabilities_cache_time) >= 300:
            _probabilities_cache = out
            _probabilities_cache_time = time.time()
            
    return _probabilities_cache


@router.get("/analytics/time_left_distribution")
async def get_time_left_distribution(db: AsyncSession = Depends(get_db_session)):
    """
    Распределение time_left_min по активам: персентили, медиана и гистограмма.
    Предназначен для подбора min_time_min / max_time_min перед переобучением.
    Кэшируется на 5 минут.
    """
    async with _time_left_dist_lock:
        global _time_left_dist_cache, _time_left_dist_cache_time
        now = time.time()
        if _time_left_dist_cache is not None and (now - _time_left_dist_cache_time) < 300:
            return _time_left_dist_cache

        stmt = select(
            MarketSnapshot.asset,
            MarketSnapshot.time_left_min,
        ).where(MarketSnapshot.final_outcome != "PENDING").order_by(MarketSnapshot.recorded_at.desc()).limit(200000)

        rows = (await db.execute(stmt)).all()
        if not rows:
            return {}

        rows_data = [{"asset": r.asset, "time_left_min": r.time_left_min} for r in rows]

        def compute_distribution(data):
            df = pd.DataFrame(data)
            bins   = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,float("inf")]
            labels = ["0m","1m","2m","3m","4m","5m","6m","7m","8m","9m",
                      "10m","11m","12m","13m","14m","15m","16m","17m","18m","19m","20m","21m",">22m"]
            result = {}
            for asset, group in df.groupby("asset"):
                t = group["time_left_min"].dropna()
                if len(t) == 0:
                    continue
                counts = (
                    pd.cut(t, bins=bins, labels=labels, right=False)
                    .value_counts()
                    .reindex(labels)
                    .fillna(0)
                )
                result[asset] = {
                    "n":              int(len(t)),
                    "median":         round(float(t.median()), 2),
                    "p10":            round(float(t.quantile(0.10)), 2),
                    "p25":            round(float(t.quantile(0.25)), 2),
                    "p75":            round(float(t.quantile(0.75)), 2),
                    "p90":            round(float(t.quantile(0.90)), 2),
                    "min":            round(float(t.min()), 2),
                    "max":            round(float(t.max()), 2),
                    "pct_above_22m":  round(float((t > 22).mean()) * 100, 1),
                    "distribution":   {k: int(v) for k, v in counts.items()},
                }
            return result

        out = await asyncio.to_thread(compute_distribution, rows_data)

        _time_left_dist_cache = out
        _time_left_dist_cache_time = time.time()

        return _time_left_dist_cache


# --- End Analytics ---
