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


def _model_type_aliases(model_type: str | None) -> list[str]:
    m = str(model_type or "logreg").strip().lower()
    if m in ("lgbm", "lightgbm"):
        return [m, "lightgbm", "lgbm"]
    if m in ("logreg", "logisticregression", "logistic_regression"):
        return [m, "logisticregression", "logreg"]
    return [m]


@dataclass
class ModelsCache:
    models: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)
    eces: dict[str, float] = field(default_factory=dict) # BUG-AO
    entries: dict[tuple[str, str, int], Any] = field(default_factory=dict)
    features_by_type: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    features_by_entry: dict[tuple[str, str, int], list[str]] = field(default_factory=dict)

    def get(self, asset: str, model_type: str = "logreg", version: int | None = None) -> Any | None:
        aliases = _model_type_aliases(model_type)
        if version is not None:
            for alias in aliases:
                if (alias, asset, version) in self.entries:
                    return self.entries[(alias, asset, version)]
        v = self.versions.get(asset)
        if v is not None:
            for alias in aliases:
                if (alias, asset, v) in self.entries:
                    return self.entries[(alias, asset, v)]
        for (mtype, a, ver), model in self.entries.items():
            if mtype in aliases and a == asset:
                return model
        return self.models.get(asset)

    def get_features(
        self,
        asset: str,
        model_type: str = "logreg",
        version: int | None = None,
    ) -> list[str]:
        aliases = _model_type_aliases(model_type)
        if version is not None:
            for alias in aliases:
                if (alias, asset, version) in self.features_by_entry:
                    return self.features_by_entry[(alias, asset, version)]
        for alias in aliases:
            if (alias, asset) in self.features_by_type:
                return self.features_by_type[(alias, asset)]
        v = self.versions.get(asset)
        if v is not None:
            for alias in aliases:
                if (alias, asset, v) in self.features_by_entry:
                    return self.features_by_entry[(alias, asset, v)]
        return self.features.get(asset, [])

    def put(
        self,
        asset: str,
        model: Any,
        model_type: str = "logreg",
        version: int = 1,
        features: list[str] | None = None,
        ece: float = 0.0,
    ) -> None:
        m_type = str(model_type or "logreg").strip().lower()
        feats = list(features) if features else []
        self.entries[(m_type, asset, version)] = model
        self.features_by_entry[(m_type, asset, version)] = feats
        self.features_by_type[(m_type, asset)] = feats
        # Also register under normalized aliases so lookup never fails
        for alias in _model_type_aliases(m_type):
            self.features_by_type[(alias, asset)] = feats
        if m_type in ("logreg", "logisticregression") or asset not in self.models:
            self.models[asset] = model
            self.versions[asset] = version
            self.features[asset] = feats
            self.eces[asset] = ece

_models_cache = None

def get_models_cache() -> ModelsCache:
    global _models_cache
    if _models_cache is None:
        _models_cache = ModelsCache(
            models={},
            versions={},
            features={},
            eces={},
            entries={},
            features_by_type={},
            features_by_entry={},
        )
    return _models_cache

def reset_models_cache() -> None:
    global _models_cache
    _models_cache = None

clear_models_cache = reset_models_cache

async def populate_models_cache(db_session: AsyncSession) -> None:
    cache = get_models_cache()
    
    # 1. Запрашиваем asset, version и model_type активных моделей
    stmt = select(ModelRegistry.asset, ModelRegistry.version, ModelRegistry.model_type).where(ModelRegistry.is_active)
    res = await db_session.execute(stmt)
    active_info = res.all()
    
    active_keys = {
        (str(row.model_type or "logreg").strip().lower(), row.asset, row.version)
        for row in active_info
    }
    active_type_keys = {(k[0], k[1]) for k in active_keys}
    for k in list(active_keys):
        for alias in _model_type_aliases(k[0]):
            active_type_keys.add((alias, k[1]))
    db_assets = {row.asset for row in active_info}
    
    # 2. Удаляем из кэша модели, которые больше не активны в базе
    for cached_key in list(cache.entries.keys()):
        if cached_key not in active_keys:
            cache.entries.pop(cached_key, None)
            cache.features_by_entry.pop(cached_key, None)

    for cached_type_key in list(cache.features_by_type.keys()):
        if cached_type_key not in active_type_keys:
            cache.features_by_type.pop(cached_type_key, None)

    for cached_asset in list(cache.models.keys()):
        if cached_asset not in db_assets:
            cache.models.pop(cached_asset, None)
            cache.versions.pop(cached_asset, None)
            cache.features.pop(cached_asset, None)
            cache.eces.pop(cached_asset, None)
            
    # 3. Находим модели, версии которых изменились или которых нет в кэше
    to_load = []
    for row in active_info:
        m_type = str(row.model_type or "logreg").strip().lower()
        if (m_type, row.asset, row.version) not in cache.entries:
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
            m_type = str(m.model_type or "logreg").strip().lower()
            cache.entries[(m_type, m.asset, m.version)] = model_obj

            m_feats = [f.strip() for f in m.features.split(",") if f.strip()] if m.features else []
            if not m_feats and hasattr(model_obj, "feature_names_in_"):
                m_feats = list(model_obj.feature_names_in_)
            cache.features_by_entry[(m_type, m.asset, m.version)] = m_feats
            cache.features_by_type[(m_type, m.asset)] = m_feats
            for alias in _model_type_aliases(m_type):
                cache.features_by_type[(alias, m.asset)] = m_feats
                cache.features_by_entry[(alias, m.asset, m.version)] = m_feats

            # LogReg takes precedence in legacy cache.models to prevent clobbering by LGBM
            if m.asset not in cache.models or m_type in ("logreg", "logisticregression"):
                cache.models[m.asset] = model_obj
                cache.versions[m.asset] = m.version
                cache.eces[m.asset] = m.ece or 0.0
                cache.features[m.asset] = m_feats

            logger.info("model_cache_updated", asset=m.asset, version=m.version, model_type=m_type)
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
        total_entries=len(cache.entries),
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
    decision_id: str | None = None,
) -> pd.DataFrame:
    """
    Строит DataFrame для инференса модели на основе исторических снапшотов и текущих (свежих) данных.
    Явно помечает decision row по decision_id.
    """
    eff_decision_id = decision_id or f"decision_{getattr(market, 'market_id', '')}_{start_time.isoformat()}"
    rows = []
    for i, snap in enumerate(history_snaps):
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
            "_row_id": str(getattr(snap, "id", None) or f"hist_{i}"),
            "_is_decision_row": False,
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
        "_row_id": eff_decision_id,
        "_is_decision_row": True,
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
        # Deterministic sort keeping decision_row traceable via _row_id
        df = df.sort_values(["recorded_at", "_is_decision_row"]).reset_index(drop=True)
        df = df.drop(columns=["recorded_at"], errors="ignore")
    if "market_id" in df.columns:
        df = df.drop(columns=["market_id"], errors="ignore")
        
    return df


def run_model_inference(
    df: pd.DataFrame,
    model: Any,
    features: list[str],
    decision_row_id: str | None = None,
) -> float:
    """
    Прогоняет DataFrame через модель и возвращает вероятность для класса 1 (flip).
    Если модель возвращает только один класс, возвращает 0.0 (или 1.0 если единственный класс - 1).
    Возвращает предсказание конкретно для decision row по decision_row_id или _is_decision_row.
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
    has_imputer = (
        hasattr(model, "named_steps") and "imputer" in getattr(model, "named_steps", {})
    )
    if not has_imputer:
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

    # Determine row index for inference
    if decision_row_id is not None and "_row_id" in df.columns:
        matching = np.where(df["_row_id"].astype(str) == str(decision_row_id))[0]
        row_idx = int(matching[0]) if len(matching) > 0 else -1
    elif "_is_decision_row" in df.columns:
        matching = np.where(df["_is_decision_row"].astype(bool))[0]
        row_idx = int(matching[0]) if len(matching) > 0 else -1
    else:
        row_idx = -1

    proba = model.predict_proba(X)

    # Determine positive class index using classes_ contract
    classes = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "classes_"):
        classes = list(model.named_steps["model"].classes_)
    elif hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        cal = model.calibrated_classifiers_[0]
        classes = getattr(cal, "classes_", None) or getattr(getattr(cal, "estimator", None), "classes_", None)
        if classes is not None:
            classes = list(classes)

    # Directly check for single-class output from proba shape
    if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] == 1:
        single_val = classes[0] if (classes and len(classes) == 1) else None
        return 1.0 if single_val in (1, True) else 0.0

    if classes is not None:
        if 1 in classes:
            pos_idx = classes.index(1)
        elif True in classes:
            pos_idx = classes.index(True)
        elif len(classes) == 1:
            return 1.0 if classes[0] in (1, True) else 0.0
        else:
            pos_idx = 1
    else:
        pos_idx = 1

    try:
        p_flip = float(proba[row_idx][pos_idx])
    except (IndexError, KeyError) as e:
        logger.warning(
            "inference_prediction_index_error",
            error=str(e),
            row_idx=row_idx,
            pos_idx=pos_idx,
            proba_shape=getattr(proba, "shape", None),
            model_type=type(model).__name__,
            classes=classes,
        )
        p_flip = 0.0

    return p_flip
