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
    deployable: dict[str, bool] = field(default_factory=dict)

_models_cache = None

def get_models_cache() -> ModelsCache:
    global _models_cache
    if _models_cache is None:
        _models_cache = ModelsCache(models={}, versions={}, features={}, eces={}, deployable={})
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
            cache.deployable.pop(cached_asset, None)

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

            is_deployable = False
            if m.is_active and getattr(m, "quality_gate_passed", False):
                meta = getattr(m, "model_metadata", {}) or {}
                if isinstance(meta, dict):
                    schema_ver = meta.get("metrics_schema_version")
                    has_oof = "oof_metrics" in meta or "oof_artifact_id" in meta
                    if schema_ver == "canonical_pnl_v1" and has_oof:
                        is_deployable = True
            cache.deployable[m.asset] = is_deployable

            m_feats = [f.strip() for f in m.features.split(",") if f.strip()] if m.features else []
            if not m_feats and hasattr(model_obj, "feature_names_in_"):
                m_feats = list(model_obj.feature_names_in_)

            cache.features[m.asset] = m_feats
            logger.info("model_cache_updated", asset=m.asset, version=m.version)
        except Exception as e:
            logger.error("Failed to load model", asset=m.asset, error=str(e))

    from polyflip.constants import PRICE_PHASE_BOUNDARIES
    _phase_suffixes = tuple(f"_{p}" for p in PRICE_PHASE_BOUNDARIES)

    phase_keys = [k for k in cache.models if k.endswith(_phase_suffixes)]
    base_keys  = [k for k in cache.models if k not in phase_keys]

    logger.info(
        "models_cache_populated",
        base_models=sorted(base_keys),
        phase_models=sorted(phase_keys),
        total=len(cache.models),
    )


def build_inference_dataframe(
    market: Any,
    history_snaps: list[Any],
    fresh_yes_price: float,
    fresh_spread: float,
    global_max: float,
    start_time: datetime,
    time_left_sec: float,
    closed_candles: list[Any] | None = None,
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
            "market_duration_min": float(getattr(snap, "market_duration_min", 15.0) or 15.0),
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
        "market_duration_min": float(getattr(market, "market_duration_min", 15.0) or 15.0),
    })

    from polyflip.models.trainer import add_derived_features
    from polyflip.models.feature_lags import add_lag_features

    df = pd.DataFrame(rows)
    df = add_derived_features(df)
    df["price_distance_from_max"] = (global_max - df["mid_price"]).clip(lower=0.0)
    df = add_lag_features(df)

    if closed_candles is not None:
        from polyflip.models.sequence_features import attach_closed_candle_features
        df = attach_closed_candle_features(
            df, closed_candles, decision_time_col="recorded_at"
        )

    if "recorded_at" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["recorded_at"]).dt.dayofweek.astype(float)
        df = df.sort_values("recorded_at").reset_index(drop=True)
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
    missing = [f for f in features if f not in df.columns]
    if missing:
        from polyflip.constants import ZERO_DEFAULT_FEATURES
        missing_required = [f for f in missing if f not in ZERO_DEFAULT_FEATURES]
        if missing_required:
            logger.error(
                "inference_feature_mismatch",
                missing=missing_required,
                available=list(df.columns),
                note="Model expects required features missing from dataframe",
            )
            raise ValueError(f"MODEL_FEATURE_MISMATCH: Missing required features: {missing_required}")

        logger.warning(
            "inference_missing_zero_default_features",
            missing=missing,
            available=list(df.columns),
            note="Filling allowed zero-default features with 0.0",
        )
        for col in missing:
            df[col] = 0.0

    X = df[features]
    non_finite = ~np.isfinite(X.astype(float).to_numpy())
    if non_finite.any():
        invalid_features = sorted(set(
            X.columns[np.flatnonzero(non_finite.any(axis=0))].tolist()
        ))
        raise ValueError(
            "MODEL_FEATURE_DATA_UNAVAILABLE: non-finite values for "
            f"{invalid_features}"
        )

    # Явная проверка порядка фич
    expected_features = None
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        base = getattr(model.calibrated_classifiers_[0], "estimator", None)
        if base and hasattr(base, "feature_names_in_"):
            expected_features = list(base.feature_names_in_)

    if expected_features is not None:
        actual_features = list(X.columns)
        if expected_features != actual_features:
            logger.error(
                "feature_order_mismatch",
                expected=expected_features,
                actual=actual_features,
                diff_missing=[f for f in expected_features if f not in actual_features],
            )
            raise ValueError(f"Feature order mismatch: expected {expected_features}, got {actual_features}")

    proba = model.predict_proba(X)

    try:
        p_flip = float(proba[-1][1])
    except IndexError:
        p_flip = 0.0

    return p_flip
