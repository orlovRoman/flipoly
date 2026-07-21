import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any
import structlog
from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pickle
from polyflip.db.models import ModelRegistry

logger = structlog.get_logger(__name__)


@dataclass
class ModelsCache:
    models: dict[str, Any]
    versions: dict[str, int]
    features: dict[str, list[str]]
    eces: dict[str, float] = field(default_factory=dict) # BUG-AO

_models_cache = None

async def get_models_cache(db_session: AsyncSession = None) -> ModelsCache:
    global _models_cache
    if _models_cache is None:
        from dataclasses import field
        _models_cache = ModelsCache(models={}, versions={}, features={}, eces={})
    
    if (not _models_cache.models or not _models_cache.versions) and db_session is not None:
        logger.warning("models_cache_empty_lazy_init")
        await populate_models_cache(db_session)
        
    return _models_cache

def clear_models_cache() -> None:
    global _models_cache
    _models_cache = None

async def populate_models_cache(db_session: AsyncSession) -> None:
    cache = get_models_cache()
    
    # 1. Запрашиваем asset и version активных моделей
    stmt = select(ModelRegistry.asset, ModelRegistry.version).where(ModelRegistry.is_active)
    res = await db_session.execute(stmt)
    active_info = res.all()
    
    db_assets = {row.asset for row in active_info}
    
    # 2. Удаляем из кэша модели, которые больше не активны в базе
    for cached_asset in list(cache.models.keys()):
        if cached_asset not in db_assets:
            cache.models.pop(cached_asset, None)
            cache.versions.pop(cached_asset, None)
            cache.features.pop(cached_asset, None)
            cache.eces.pop(cached_asset, None) # BUG-AO
            
    # 3. Находим модели, версии которых изменились или которых нет в кэше
    to_load = []
    for row in active_info:
        cached_ver = cache.versions.get(row.asset)
        if cached_ver is None or cached_ver != row.version:
            to_load.append(row.asset)
            
    if not to_load:
        return
        
    # 4. Загружаем изменившиеся/новые модели
    load_stmt = select(ModelRegistry).where(
        ModelRegistry.is_active,
        ModelRegistry.asset.in_(to_load)
    )
    models_to_load = (await db_session.execute(load_stmt)).scalars().all()
    
    for m in models_to_load:
        try:
            model_obj = pickle.loads(m.model_blob)
            cache.models[m.asset] = model_obj
            cache.versions[m.asset] = m.version
            cache.eces[m.asset] = m.ece or 0.0 # BUG-AO
            
            m_feats = [f.strip() for f in m.features.split(",") if f.strip()] if m.features else []
            if not m_feats and hasattr(model_obj, "feature_names_in_"):
                m_feats = list(model_obj.feature_names_in_)
                
            cache.features[m.asset] = m_feats
            logger.info("model_cache_updated", asset=m.asset, version=m.version)
        except Exception as e:
            logger.error("Failed to load model", asset=m.asset, error=str(e))


def build_inference_dataframe(
    market: Any,
    history_snaps: list[Any],
    fresh_yes_price: float,
    fresh_spread: float,
    global_max: float,
    start_time: datetime,
    time_left_sec: float,
) -> pd.DataFrame:
    """
    Строит DataFrame для инференса модели на основе исторических снапшотов и текущих (свежих) данных.
    """
    rows = []
    for snap in history_snaps:
        rows.append({
            "time_left_min": getattr(snap, "time_left_min", 0.0),
            "mid_price": getattr(snap, "mid_price", 0.0),
            "spread": getattr(snap, "spread", 0.0),
            "price_velocity": getattr(snap, "price_velocity", 0.0),
            "volume_5min": getattr(snap, "volume_5min", 0.0),
            "hour_of_day": getattr(snap, "hour_of_day", 0),
            "market_id": getattr(snap, "market_id", ""),
            "recorded_at": getattr(snap, "recorded_at", None),
        })
        
    rows.append({
        "time_left_min": time_left_sec / 60.0,
        "mid_price": fresh_yes_price,
        "spread": fresh_spread,
        "price_velocity": getattr(market, "price_velocity", 0.0) or 0.0,
        "volume_5min": getattr(market, "volume_5min", 0.0) or 0.0,
        "hour_of_day": start_time.hour,
        "market_id": getattr(market, "market_id", ""),
        "recorded_at": start_time,
    })
    
    from polyflip.models.trainer import add_derived_features
    from polyflip.models.feature_lags import add_lag_features

    df = pd.DataFrame(rows)
    df = add_derived_features(df)
    df["price_distance_from_max"] = (global_max - df["mid_price"]).clip(lower=0.0)
    df = add_lag_features(df)
    
    if "recorded_at" in df.columns:
        df = df.drop(columns=["recorded_at"], errors="ignore")
    if "market_id" in df.columns:
        df = df.drop(columns=["market_id"], errors="ignore")
        
    return df


def run_model_inference(
    df: pd.DataFrame,
    model: Any,
    features: list[str],
) -> float:
    """
    Прогоняет DataFrame через модель и возвращает вероятность для класса 1 (flip).
    Если модель возвращает только один класс, возвращает 0.0.
    """
    X = df[features]
    proba = model.predict_proba(X)
    
    try:
        p_flip = float(proba[-1][1])
    except IndexError:
        p_flip = 0.0
        
    return p_flip
