import pickle
import numpy as np
import pandas as pd
import asyncio
import functools
from datetime import datetime, timedelta, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, precision_recall_curve

from polyflip.db.models import CryptoCandle, MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.config import settings
from polyflip.constants import (
    ASSET_TO_BINANCE_SYMBOL,
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    MODEL_THRESHOLD_MIN,
    MODEL_THRESHOLD_MAX,
)

_TRAINING_LOCKS: dict[str, asyncio.Lock] = {}
_TRAINING_SEMAPHORE: asyncio.Semaphore | None = None
logger = structlog.get_logger(__name__)


async def _get_training_semaphore(db: AsyncSession) -> asyncio.Semaphore:
    """
    Возвращает синглтон-семафор для лимита параллельных вызовов обучения.
    ВНИМАНИЕ: Изменение TRAIN_MAX_PARALLEL_JOBS в RuntimeSettings вступает в силу
    после перезапуска сервиса, так как Семафор инициализируется один раз при старте.
    """
    global _TRAINING_SEMAPHORE
    if _TRAINING_SEMAPHORE is None:
        from polyflip.services.settings_service import get_int
        max_jobs = await get_int(db, "TRAIN_MAX_PARALLEL_JOBS") or 2
        _TRAINING_SEMAPHORE = asyncio.Semaphore(max_jobs)
    return _TRAINING_SEMAPHORE

from polyflip.models.feature_lags import add_lag_features, LAG_FEATURE_NAMES
from polyflip.models.sequence_features import (
    FEATURE_EXPERIMENT_LABELS,
    FEATURE_EXPERIMENT_VARIANTS,
    SEQUENCE_DIRECTION_FEATURES,
    SEQUENCE_CANDLE_FEATURES,
    normalize_experiment_variant,
    SEQUENCE_FEATURE_SET_VERSION,
    attach_closed_candle_features,
    sequence_history_ready,
)
from polyflip.models.temporal_validation import (
    grouped_walk_forward_folds,
    latest_group_holdout,
    market_balanced_weights,
)

DERIVED_FEATURES = [
    "price_deviation",
    "spread_pct",
    "log_time_left",
    "day_of_week",
    "price_distance_from_max",
    "is_final_phase",
    "high_price_final",
    *LAG_FEATURE_NAMES,
]

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_deviation"]     = (df["mid_price"] - 0.5).abs()
    df["spread_pct"]          = (df["spread"] / (df["mid_price"] + 1e-6)).clip(upper=10.0)
    df["log_time_left"]       = np.log1p(df["time_left_min"])
    df["deviation_x_time"] = df["price_deviation"] * df["time_left_min"]
    df["price_deviation_sq"] = df["price_deviation"] ** 2

    if "day_of_week" not in df.columns:
        if "recorded_at" in df.columns:
            df["day_of_week"] = pd.to_datetime(df["recorded_at"]).dt.weekday.astype(float)
        else:
            df["day_of_week"] = 0.0

    # price_distance_from_max: по всей истории рынка в датасете
    if "market_id" in df.columns and "recorded_at" in df.columns:
        # expanding max по времени внутри рынка
        df_sorted = df.sort_values(["market_id", "recorded_at"])
        expanding_max = (
            df_sorted.groupby("market_id")["mid_price"]
            .transform(lambda x: x.expanding().max())
        )
        # Присваиваем через индекс, а не .values — это безопасно
        df["price_distance_from_max"] = (
            expanding_max - df_sorted["mid_price"]
        ).clip(lower=0.0).reindex(df.index)
    elif "market_id" in df.columns:
        df["_market_max"] = df.groupby("market_id")["mid_price"].transform("max")
        df["price_distance_from_max"] = (df["_market_max"] - df["mid_price"]).clip(lower=0.0)
        df.drop(columns=["_market_max"], inplace=True)
    else:
        df["price_distance_from_max"] = 0.0

    if "market_duration_min" in df.columns:
        denominator = df["market_duration_min"].clip(lower=15.0)
        time_phase = (df["time_left_min"] / (denominator + 1e-6)).clip(0, 1)
    elif "market_id" in df.columns and "time_left_min" in df.columns:
        if len(df) > df["market_id"].nunique():
            denominator = (
                df.groupby("market_id")["time_left_min"].transform("max")
                .clip(lower=15.0)
            )
            time_phase = (df["time_left_min"] / (denominator + 1e-6)).clip(0, 1)
        else:
            time_phase = (df["time_left_min"] / 15.0).clip(0, 1)
    elif "time_left_min" in df.columns:
        time_phase = (df["time_left_min"] / 15.0).clip(0, 1)
    else:
        time_phase = 1.0

    # --- Interaction Features ---
    # Preserve the exact schema used by legacy LogReg artifacts. These are
    # calculated for compatibility but are not auto-added to new models.
    df["time_phase"] = time_phase
    velocity = (
        df["price_velocity"].fillna(0.0)
        if "price_velocity" in df.columns
        else 0.0
    )
    df["velocity_x_phase"] = velocity * (1.0 - time_phase)
    df["dev_sq_x_phase"] = df["price_deviation_sq"] * (1.0 - time_phase)

    df["is_final_phase"] = (time_phase <= 0.20).astype(float)
    df["high_price_final"] = df["price_deviation"] * (1.0 - time_phase)

    return df

def _compute_sample_weights(
    time_left: np.ndarray,
    mode: str,
    tau: float = 5.0,
) -> np.ndarray | None:
    """
    Возвращает массив весов для LogisticRegression.fit(model__sample_weight=...).
    'uniform'    → None (sklearn использует равные веса)
    'time_decay' → 1 / (time_left + 1)   — простой обратный вес
    'exp_decay'  → exp(-time_left / tau)  — экспоненциальный с параметром tau
    """
    if mode == "uniform":
        return None
    if mode == "time_decay":
        w = 1.0 / (time_left + 1.0)
    elif mode == "exp_decay":
        w = np.exp(-time_left / (tau + 1e-9))
    else:
        logger.warning("unknown_weight_mode", mode=mode, fallback="uniform")
        return None
    # Нормализуем: среднее = 1.0 (не меняет масштаб градиента)
    w = w / (w.mean() + 1e-9)
    return w.astype(np.float64)


def _compute_backtest_pnl(
    oof_scores: np.ndarray,
    y: pd.Series,
    mid_prices: pd.Series,
    threshold: float,
    fee_per_trade: float = 0.02,
    stake: float = 1.0,
) -> dict:
    outsider_signals = oof_scores >= threshold
    if outsider_signals.sum() == 0:
        return {
            "total_pnl": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "sharpe": None, "strategy_branch": "OUTSIDER_ONLY",
        }
    
    # flip_vs_final=True means the current favourite loses: buy the outsider.
    prices = np.minimum(mid_prices.values[outsider_signals], 1.0 - mid_prices.values[outsider_signals])
    targets = y.values[outsider_signals]
    trade_pnl = np.where(
        targets == 1,
        (1.0 - prices) * stake - fee_per_trade,
        -prices * stake - fee_per_trade,
    )
    total_pnl = float(trade_pnl.sum())
    n_trades = int(outsider_signals.sum())
    win_rate = float((trade_pnl > 0).mean())
    avg_pnl = float(trade_pnl.mean())
    if n_trades > 1 and trade_pnl.std() > 1e-9:
        sharpe = float(trade_pnl.mean() / trade_pnl.std() * np.sqrt(n_trades))
    else:
        sharpe = None
    return {
        "total_pnl": round(total_pnl, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_trade_pnl": round(avg_pnl, 4),
        "sharpe": round(sharpe, 4) if sharpe else None,
        "strategy_branch": "OUTSIDER_ONLY",
    }


def _group_holdout_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    timestamps: pd.Series | None,
    *,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if timestamps is not None:
        return latest_group_holdout(
            groups, timestamps, validation_fraction=validation_fraction
        )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=CV_RANDOM_STATE,
    )
    return next(splitter.split(X, y, groups=groups))


def _outer_validation_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    timestamps: pd.Series | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    if timestamps is None:
        n_splits = min(CV_N_SPLITS, int(groups.nunique()))
        if n_splits < 2:
            return [], []
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y, groups=groups)), []

    folds = grouped_walk_forward_folds(
        groups, timestamps, n_splits=CV_N_SPLITS
    )
    metadata = [
        {
            "train_markets": len(fold.train_groups),
            "validation_markets": len(fold.validation_groups),
            "train_end": fold.train_end.isoformat(),
            "validation_start": fold.validation_start.isoformat(),
            "validation_end": fold.validation_end.isoformat(),
        }
        for fold in folds
    ]
    return [
        (fold.train_index, fold.validation_index) for fold in folds
    ], metadata


def _fit_and_serialize(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    mid_prices: pd.Series,
    sample_weight: np.ndarray | None = None,
    lr_coef_threshold: float = 0.005,
    lr_min_features: int = 4,
    min_precision: float = 0.52,
    timestamps: pd.Series | None = None,
    max_suspicious: float = 0.95,
    fee_per_trade: float = 0.02,
):
    """Синхронная CPU-bound функция для кросс-валидации, обучения и сериализации модели."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    mid_prices = mid_prices.reset_index(drop=True)
    if timestamps is not None:
        timestamps = pd.Series(timestamps).reset_index(drop=True)
    sample_weight = market_balanced_weights(groups, sample_weight)

    if sample_weight is not None:
        logger.info(
            "sample_weights_distribution",
            w_min=round(float(sample_weight.min()), 4),
            w_max=round(float(sample_weight.max()), 4),
            w_mean=round(float(sample_weight.mean()), 4),
            w_p10=round(float(np.percentile(sample_weight, 10)), 4),
            w_p90=round(float(np.percentile(sample_weight, 90)), 4),
            ratio_p90_p10=round(
                float(np.percentile(sample_weight, 90)) /
                (float(np.percentile(sample_weight, 10)) + 1e-9), 2
            ),
        )

    # --- Grid search по C (оптимизировано: 1 сплит GroupShuffleSplit) ---
    C_GRID = [0.1, 0.5, 1.0, 5.0]
    c_results = {}

    try:
        tr_idx, vl_idx = _group_holdout_indices(
            X, y, groups, timestamps,
            validation_fraction=0.25,
        )
        if len(np.unique(y.iloc[tr_idx])) >= 2 and len(np.unique(y.iloc[vl_idx])) >= 2:
            m_weight_tr = sample_weight[tr_idx] if sample_weight is not None else None
            for c_val in C_GRID:
                probe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(
                        class_weight="balanced", C=c_val,
                        random_state=CV_RANDOM_STATE, max_iter=300,
                        solver="lbfgs", n_jobs=1,
                    )),
                ])
                probe.fit(X.iloc[tr_idx], y.iloc[tr_idx], model__sample_weight=m_weight_tr)
                proba = probe.predict_proba(X.iloc[vl_idx])[:, 1]
                c_results[c_val] = round(float(roc_auc_score(y.iloc[vl_idx], proba)), 4)
    except Exception as e:
        logger.warning("c_grid_search_fallback", error=str(e))

    best_C = max(c_results, key=c_results.get) if c_results else 1.0
    logger.info("c_grid_search_results", c_grid=c_results, best_C=best_C)

    # 3. Обучаем модель с кросс-валидацией
    validation_splits, validation_fold_metadata = _outer_validation_splits(
        X, y, groups, timestamps
    )
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", C=best_C,
            random_state=CV_RANDOM_STATE, max_iter=300,
            solver="lbfgs", n_jobs=1,
        ))
    ])
    
    from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
    
    aucs = []
    oof_scores = np.full(len(y), np.nan, dtype=float)
    for train_index, val_index in validation_splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        
        # Split the outer training markets again: the calibrator must never see
        # rows used to fit the base estimator.
        inner_groups = groups.iloc[train_index].reset_index(drop=True)
        inner_timestamps = (
            timestamps.iloc[train_index].reset_index(drop=True)
            if timestamps is not None else None
        )
        base_idx, calibration_idx = _group_holdout_indices(
            X_train, y_train, inner_groups, inner_timestamps,
            validation_fraction=0.2,
        )
        fold_base = clone(base_model)
        if (
            len(np.unique(y_train.iloc[base_idx])) < 2
            or len(np.unique(y_train.iloc[calibration_idx])) < 2
        ):
            tr_weight = sample_weight[train_index] if sample_weight is not None else None
            fold_base.fit(X_train, y_train, model__sample_weight=tr_weight)
            y_proba = fold_base.predict_proba(X_val)[:, 1]
        else:
            tr_weight = (
                sample_weight[train_index][base_idx]
                if sample_weight is not None else None
            )
            fold_base.fit(
                X_train.iloc[base_idx], y_train.iloc[base_idx],
                model__sample_weight=tr_weight,
            )
            fold_calib = CalibratedClassifierCV(
                estimator=FrozenEstimator(fold_base), method="sigmoid", cv=None
            )
            fold_calib.fit(
                X_train.iloc[calibration_idx], y_train.iloc[calibration_idx]
            )
            y_proba = fold_calib.predict_proba(X_val)[:, 1]
        oof_scores[val_index] = y_proba
        aucs.append(roc_auc_score(y_val, y_proba))
        
    val_acc = float(np.mean(aucs)) if aucs else 0.5
    
    # Baseline ROC-AUC/Accuracy (доля мажоритарного класса)
    valid_oof_mask = np.isfinite(oof_scores)
    if not valid_oof_mask.any():
        return None
    valid_y = y.to_numpy()[valid_oof_mask]
    valid_scores = oof_scores[valid_oof_mask]

    baseline_acc = 0.5
    
    # ECE Diagnostic по откалиброванным предсказаниям
    bin_ids = np.minimum((valid_scores * 10).astype(int), 9)
    ece = 0.0
    for bin_id in range(10):
        in_bin = bin_ids == bin_id
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(valid_y[in_bin].mean()) - float(valid_scores[in_bin].mean())
            )
    logger.info("calibration_check", ece=round(ece, 4))
    
    # Обучаем финальную модель на всех данных (с holdout для честной калибровки)
    # Чтобы исключить Group Leakage (BUG-05), разбиваем выборку по группам (market_id)
    train_idx, cal_idx = _group_holdout_indices(
        X, y, groups, timestamps,
        validation_fraction=0.2,
    )
    X_train_cal, X_cal = X.iloc[train_idx], X.iloc[cal_idx]
    y_train_cal, y_cal = y.iloc[train_idx], y.iloc[cal_idx]
    
    if len(np.unique(y_train_cal)) < 2 or len(np.unique(y_cal)) < 2:
        # Fallback to uncalibrated model on entire dataset if split is invalid
        final_model = clone(base_model)
        try:
            sw_all = sample_weight if sample_weight is not None else None
            final_model.fit(X, y, model__sample_weight=sw_all)
        except Exception:
            return None # Impossible to fit
        final_base = final_model
    else:
        final_base = clone(base_model)
        tr_cal_weight = sample_weight[train_idx] if sample_weight is not None else None
        final_base.fit(X_train_cal, y_train_cal, model__sample_weight=tr_cal_weight)
        
        final_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(final_base),
            method="sigmoid",
            cv=None
        )
        final_model.fit(X_cal, y_cal)
    
    coefs = final_base.named_steps["model"].coef_[0]
    coef_info = dict(zip(list(X.columns), [round(float(c), 4) for c in coefs]))
    logger.info("model_feature_weights", coefficients=coef_info)
    
    # --- ДОБАВИТЬ: ранжирование по |coef| ---
    abs_coefs = sorted(
        [(feat, abs(float(c))) for feat, c in zip(X.columns, coefs)],
        key=lambda x: x[1], reverse=True,
    )
    logger.info(
        "feature_importance_top10",
        top_features=[{"feature": f, "abs_coef": round(v, 4)} for f, v in abs_coefs[:10]],
        bottom_features=[{"feature": f, "abs_coef": round(v, 4)} for f, v in abs_coefs[-5:]],
    )

    weak_features = [f for f, v in abs_coefs if v < lr_coef_threshold]
    if len(X.columns) - len(weak_features) >= lr_min_features and weak_features:
        logger.warning(
            "weak_features_detected",
            count=len(weak_features),
            features=weak_features,
            threshold=lr_coef_threshold,
            suggestion="Consider removing from ACTIVE_FEATURES via dashboard",
        )
    
    # Калибровка порога с использованием Out-Of-Fold предсказаний (исключаем Data Leakage)
    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(valid_y, valid_scores)

    # Найти порог с лучшим F1 среди тех где precision >= min_precision
    valid_mask = precision_arr[:-1] >= min_precision
    if valid_mask.any():
        f1 = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
        f1_filtered = np.where(valid_mask, f1, 0)
        optimal_threshold = float(thresholds_pr[np.argmax(f1_filtered)])
    else:
        f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
        if len(thresholds_pr) > 0:
            optimal_threshold = float(thresholds_pr[np.argmax(f1_scores)])
        else:
            optimal_threshold = 0.65

    # Проверка на leakage временно отключена, т.к. для decided-рынков
    # порог может легально достигать 1.0 (сигнал сильный).
    if optimal_threshold >= max_suspicious:
        logger.warning("suspicious_threshold", threshold=optimal_threshold, max=max_suspicious)

    best_thr_idx = np.searchsorted(thresholds_pr, optimal_threshold - 1e-9)
    best_thr_idx = min(best_thr_idx, len(precision_arr) - 2)
    _prec = float(precision_arr[best_thr_idx])
    _rec  = float(recall_arr[best_thr_idx])
    _f1   = 2 * _prec * _rec / (_prec + _rec + 1e-8)
    
    logger.info(
        "threshold_diagnostics",
        optimal_threshold=round(optimal_threshold, 4),
        precision=round(_prec, 4),
        recall=round(_rec, 4),
        f1=round(_f1, 4),
        val_auc=round(val_acc, 4),
        baseline_auc=round(baseline_acc, 4),
        ece=round(ece, 4),
        min_precision_used=min_precision,
        n_samples=len(y),
        fold_aucs=[round(a, 4) for a in aucs],
    )

    # Сериализуем модель (Pipeline сохраняет скейлер внутри)
    model_bytes = pickle.dumps(final_model)
    
    backtest = _compute_backtest_pnl(
        oof_scores=valid_scores,
        y=pd.Series(valid_y),
        mid_prices=pd.Series(mid_prices.to_numpy()[valid_oof_mask]),
        threshold=optimal_threshold,
        fee_per_trade=fee_per_trade,
    )
    backtest.update({
        "validation_scheme": (
            "GROUPED_WALK_FORWARD" if timestamps is not None else "GROUP_K_FOLD"
        ),
        "validation_folds": validation_fold_metadata,
        "oot_samples": int(valid_oof_mask.sum()),
        "oot_markets": int(pd.Series(
            groups.to_numpy()[valid_oof_mask]
        ).nunique()),
        "brier_score": round(float(brier_score_loss(valid_y, valid_scores)), 6),
        "log_loss": round(float(log_loss(valid_y, valid_scores, labels=[0, 1])), 6),
        "model_config": {"penalty": "l2", "solver": "lbfgs", "C": best_C},
        "c_search_auc": {str(key): value for key, value in c_results.items()},
        "market_balanced_weights": True,
    })

    logger.info(
        "backtest_pnl_result",
        total_pnl=backtest["total_pnl"],
        n_trades=backtest["n_trades"],
        win_rate=backtest["win_rate"],
        sharpe=backtest["sharpe"],
    )

    return model_bytes, val_acc, baseline_acc, optimal_threshold, ece, backtest
def serialize_training(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        asset = kwargs.get("asset")
        if not asset and len(args) > 1:
            asset = args[1]
        if not asset:
            asset = "__global__"
            
        lock = _TRAINING_LOCKS.setdefault(asset, asyncio.Lock())
            
        if lock.locked():
            logger.warning("train_model_queued", asset=asset, note="Training is already running. Waiting in queue...")
            
        async with lock:
            return await func(*args, **kwargs)
    return wrapper

class ModelTrainer:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.status_messages = {}

    @serialize_training
    async def train_model(
        self, asset: str, save_settings: bool = True, feature_set: str = "AUTO"
    ) -> bool:
        """
        Обучает модель LogisticRegression для заданного актива на основе 
        исторических (разрезолвленных) данных и сохраняет в БД.
        """
        logger.info("starting_training", asset=asset)
        
        # Получаем активные фичи из RuntimeSettings
        experiment_variant = normalize_experiment_variant(feature_set)
        settings_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "ACTIVE_FEATURES")
        settings_result = await self.db.execute(settings_stmt)
        active_features_setting = settings_result.scalar_one_or_none()
        
        if active_features_setting and active_features_setting.value.strip():
            active_features = active_features_setting.value.split(",")
        else:
            active_features = settings.ACTIVE_FEATURES.split(",")
            
        active_features = [f.strip() for f in active_features if f.strip()]
        
        if experiment_variant in FEATURE_EXPERIMENT_VARIANTS:
            active_features = [
                feature for feature in active_features
                if feature not in SEQUENCE_CANDLE_FEATURES
            ]
        if not active_features:
            logger.error("no_active_features_selected", asset=asset)
            self.status_messages[asset] = "Ошибка: не выбраны активные признаки"
            return False
        
        # 1. Сначала проверяем количество доступных сэмплов через быстрый COUNT(*)
        from polyflip.services.settings_service import get_float, get_int, get_setting
        min_time_min = await get_float(self.db, "LR_TRAIN_MIN_TIME_LEFT_MIN")
        max_time_min = await get_float(self.db, "LR_TRAIN_MAX_TIME_LEFT_MIN")

        count_stmt = select(func.count(MarketSnapshot.id)).where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.final_outcome.in_(["YES", "NO"]),
            MarketSnapshot.flip_vs_final.is_not(None),
            MarketSnapshot.time_left_min >= min_time_min,
            MarketSnapshot.time_left_min <= max_time_min
        )
        count_result = await self.db.execute(count_stmt)
        total_samples = count_result.scalar() or 0
        
        # BUG-004 FIX: Используем настройку из конфига
        if total_samples < settings.MIN_SAMPLES_FOR_MODEL:
            logger.warning("not_enough_data_for_training", asset=asset, samples=total_samples, required=settings.MIN_SAMPLES_FOR_MODEL)
            self.status_messages[asset] = f"Пропущено: недостаточно данных ({total_samples}/{settings.MIN_SAMPLES_FOR_MODEL})"
            return False

        # Получаем обучающую выборку (только YES и NO, и где есть рассчитанный флип)
        stmt = select(MarketSnapshot).where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.final_outcome.in_(["YES", "NO"]),
            MarketSnapshot.flip_vs_final.is_not(None),
            MarketSnapshot.time_left_min >= min_time_min,
            MarketSnapshot.time_left_min <= max_time_min
        )
        result = await self.db.execute(stmt)
        snapshots = result.scalars().all()

        # 2. Формируем DataFrame
        data = []
        target_mismatches = 0
        for s in snapshots:
            if s.recorded_at is None:
                target_mismatches += 1
                continue
            expected_flip = (
                False
                if float(s.mid_price) == 0.5
                else ((float(s.mid_price) > 0.5) != (s.final_outcome == "YES"))
            )
            if bool(s.flip_vs_final) != expected_flip:
                target_mismatches += 1
                continue

            data.append({
                "market_id": s.market_id,
                "recorded_at": s.recorded_at,
                "time_left_min": s.time_left_min,
                "mid_price": s.mid_price,
                "spread": s.spread,
                "price_velocity": s.price_velocity,
                "volume_5min": s.volume_5min,
                "hour_of_day": s.hour_of_day,
                "day_of_week": float(s.recorded_at.weekday()) if s.recorded_at else 0.0,
                "target": 1 if s.flip_vs_final else 0
            })
            
        df = pd.DataFrame(data)
        
        if target_mismatches:
            logger.warning(
                "training_rows_rejected_by_target_contract",
                asset=asset,
                rejected=target_mismatches,
                target_source="POLYMARKET_FLIP_VS_FINAL_OUTCOME",
            )
        df = (
            df.drop_duplicates(["market_id", "recorded_at"], keep="last")
            .sort_values(["market_id", "recorded_at"])
            .reset_index(drop=True)
        )
        if df.empty:
            self.status_messages[asset] = "Training failed: no target-consistent rows"
            return False

        sequence_coverage = 0.0
        sequence_enabled = False
        sequence_symbol = ASSET_TO_BINANCE_SYMBOL.get(asset.split("_")[0])
        if sequence_symbol:
            candle_start = pd.Timestamp(df["recorded_at"].min()).to_pydatetime() - timedelta(days=2)
            candle_end = pd.Timestamp(df["recorded_at"].max()).to_pydatetime()
            candle_stmt = (
                select(CryptoCandle)
                .where(
                    CryptoCandle.symbol == sequence_symbol,
                    CryptoCandle.interval == "15m",
                    CryptoCandle.is_closed.is_(True),
                    CryptoCandle.close_time.is_not(None),
                    CryptoCandle.close_time >= candle_start,
                    CryptoCandle.close_time <= candle_end,
                )
                .order_by(CryptoCandle.close_time.asc())
            )
            candle_rows = (await self.db.execute(candle_stmt)).scalars().all()
            df = attach_closed_candle_features(df, candle_rows)
            ready_mask = sequence_history_ready(df)
            sequence_coverage = float(ready_mask.mean()) if len(df) else 0.0
            sequence_enabled = sequence_coverage >= 0.80
            if sequence_enabled:
                df = df.loc[ready_mask].reset_index(drop=True)
            logger.info(
                "closed_candle_sequence_coverage",
                asset=asset,
                symbol=sequence_symbol,
                candles=len(candle_rows),
                coverage=round(sequence_coverage, 4),
                enabled=sequence_enabled,
                feature_set=SEQUENCE_FEATURE_SET_VERSION,
            )

        requested_sequence = set(active_features) & set(SEQUENCE_CANDLE_FEATURES)
        if requested_sequence and not sequence_enabled:
            self.status_messages[asset] = (
                "Training failed: closed-candle sequence history coverage "
                f"{sequence_coverage:.1%} is below 80%"
            )
            return False

        if len(df) < settings.MIN_SAMPLES_FOR_MODEL:
            self.status_messages[asset] = (
                f"Training failed: {len(df)} valid rows remain after data contracts"
            )
            return False


        if not df.empty:
            logger.info("time_left_distribution", 
                asset=asset,
                n_snapshots=len(df),
                min_min=round(df["time_left_min"].min(), 2),
                max_min=round(df["time_left_min"].max(), 2),
                median_min=round(df["time_left_min"].median(), 2),
                p25=round(df["time_left_min"].quantile(0.25), 2),
                p75=round(df["time_left_min"].quantile(0.75), 2),
                n_markets=df["market_id"].nunique(),
                snapshots_per_market=round(len(df) / max(df["market_id"].nunique(), 1), 1),
            )

        # Добавляем инженерные признаки
        df = add_derived_features(df)
        df = add_lag_features(df)
        df["_decision_at"] = pd.to_datetime(df["recorded_at"], utc=True)
        df.drop(columns=["recorded_at"], errors="ignore", inplace=True)

        # Автоматически расширяем active_features производными признаками,
        # если их базовые источники (mid_price, spread, time_left_min) присутствуют
        base_for_derived = {"mid_price", "spread", "time_left_min"}
        if base_for_derived.issubset(set(active_features)):
            generated_features = list(DERIVED_FEATURES)
            if experiment_variant in FEATURE_EXPERIMENT_VARIANTS:
                generated_features.extend(
                    FEATURE_EXPERIMENT_VARIANTS[experiment_variant]
                )
            elif sequence_enabled:
                generated_features.extend(SEQUENCE_CANDLE_FEATURES)
            for feat in generated_features:
                if feat not in active_features:
                    active_features.append(feat)
            logger.info("derived_features_added", features=generated_features, asset=asset)
            
            # Синхронизируем расширенный список с БД RuntimeSettings (без принудительной молчаливой перезаписи)
            derived_setting = await self.db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == "ACTIVE_FEATURES")
            )
            derived_row = derived_setting.scalar_one_or_none()
            if derived_row:
                new_value = ",".join(active_features)
                if derived_row.value != new_value:
                    op_features = set(derived_row.value.split(","))
                    trainer_features = set(new_value.split(","))
                    silently_added = trainer_features - op_features
                    silently_removed = op_features - trainer_features
                    logger.warning(
                        "active_features_operator_setting_preserved",
                        asset=asset,
                        silently_added=sorted(silently_added),
                        silently_removed=sorted(silently_removed),
                        note=(
                            "Trainer wanted to change ACTIVE_FEATURES but operator setting "
                            "was preserved. Update via dashboard if needed."
                        ),
                    )
        
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
            
        # Гарантируем позиционные индексы — критично для корректной
        # индексации sample_weight[train_idx] внутри _fit_and_serialize
        df = df.reset_index(drop=True)

        X = df[active_features]
        y = df["target"]
        groups = df["market_id"]

        import hashlib
        fingerprint_rows = pd.util.hash_pandas_object(
            pd.concat(
                [df[["market_id", "target", "_decision_at"]], X],
                axis=1,
            ),
            index=False,
        ).values.tobytes()
        fingerprint_meta = (
            f"{asset}|n={len(df)}|features={','.join(active_features)}"
        ).encode()
        dataset_fingerprint = hashlib.md5(
            fingerprint_meta + fingerprint_rows
        ).hexdigest()

        lr_coef_threshold = await get_float(self.db, "LR_COEF_THRESHOLD")
        lr_min_features = await get_int(self.db, "LR_MIN_FEATURES")
        min_precision = await get_float(self.db, "LGBM_MIN_PRECISION_FOR_THRESHOLD")
        max_suspicious = await get_float(self.db, "LGBM_MAX_SUSPICIOUS_THRESHOLD")
        weight_mode = await get_setting(self.db, "LR_SAMPLE_WEIGHT_MODE")
        weight_tau = await get_float(self.db, "LR_SAMPLE_WEIGHT_TAU")

        sample_weights = _compute_sample_weights(
            time_left=df["time_left_min"].values,
            mode=weight_mode,
            tau=weight_tau,
        )

        if sample_weights is not None:
            assert len(sample_weights) == len(X), (
                f"sample_weights size mismatch: "
                f"{len(sample_weights)} != {len(X)}. "
                f"Убедись что df.reset_index(drop=True) вызван перед X = df[FEATURE_COLS]"
            )

        # Выполняем CPU-bound обучение в отдельном потоке с лимитом параллелизма (Semaphore)
        fee_per_trade = await get_float(self.db, "BACKTEST_FEE_PER_TRADE")
        sem = await _get_training_semaphore(self.db)
        async with sem:
            fit_res = await asyncio.to_thread(
                _fit_and_serialize, X, y, groups,
                mid_prices=df["mid_price"],
                sample_weight=sample_weights,
                lr_coef_threshold=lr_coef_threshold,
                lr_min_features=lr_min_features,
                timestamps=df["_decision_at"],
                min_precision=min_precision,
                max_suspicious=max_suspicious,
                fee_per_trade=fee_per_trade,
            )
        if fit_res is None:
            logger.error("model_fit_failed", asset=asset)
            self.status_messages[asset] = "Training failed: no valid OOF predictions"
            return False
        model_bytes, val_acc, baseline_acc, optimal_threshold, ece, backtest = fit_res

        logger.info("model_trained", asset=asset, samples=len(df), val_auc=val_acc, baseline_auc=baseline_acc, ece=ece)

        # --- Model Quality Gate Check ---
        lift = val_acc - baseline_acc
        max_lift_loss = -0.005  # accuracy не должна быть ниже baseline более чем на 0.5%

        active_model_stmt = (
            select(ModelRegistry)
            .where(ModelRegistry.asset == asset, ModelRegistry.is_active == True)
            .limit(1)
        )
        active_res = await self.db.execute(active_model_stmt)
        active_model = active_res.scalar_one_or_none()

        passed_quality_gate = True
        gate_reasons = []

        if lift < max_lift_loss:
            passed_quality_gate = False
            gate_reasons.append(f"Negative lift vs baseline: {lift:+.4f} (accuracy={val_acc:.4f}, baseline={baseline_acc:.4f})")

        if ece > 0.15:
            passed_quality_gate = False
            gate_reasons.append(f"Excessive ECE calibration error: {ece:.4f} > 0.15")

        if active_model is not None and active_model.accuracy is not None:
            same_dataset = (
                hasattr(active_model, "dataset_fingerprint")
                and active_model.dataset_fingerprint == dataset_fingerprint
            )
            if same_dataset:
                acc_diff = val_acc - active_model.accuracy
                if acc_diff < -0.02:
                    passed_quality_gate = False
                    gate_reasons.append(f"Accuracy degraded vs active model v{active_model.version}: {acc_diff:+.4f} < -0.02 (same dataset)")
            else:
                logger.warning(
                    "quality_gate_dataset_changed",
                    asset=asset,
                    old_fingerprint=getattr(active_model, "dataset_fingerprint", "none"),
                    new_fingerprint=dataset_fingerprint,
                    note="AUC comparison skipped — dataset changed. Using baseline check only."
                )

        if optimal_threshold < MODEL_THRESHOLD_MIN or optimal_threshold > MODEL_THRESHOLD_MAX:
            passed_quality_gate = False
            gate_reasons.append(
                f"Threshold {optimal_threshold:.4f} outside advisory bounds [{MODEL_THRESHOLD_MIN}, {MODEL_THRESHOLD_MAX}]"
            )

        # Новая проверка: backtested PnL
        MIN_BACKTEST_TRADES = await get_int(self.db, "BACKTEST_MIN_TRADES")
        MIN_BACKTEST_PNL = await get_float(self.db, "BACKTEST_MIN_PNL")
        if backtest["n_trades"] >= MIN_BACKTEST_TRADES:
            if backtest["total_pnl"] < MIN_BACKTEST_PNL:
                passed_quality_gate = False
                gate_reasons.append(
                    f"Backtest PnL negative: {backtest['total_pnl']:.4f} "
                    f"on {backtest['n_trades']} trades "
                    f"(WR={backtest['win_rate']:.1%})"
                )
        else:
            logger.warning(
                "backtest_gate_skipped",
                asset=asset,
                n_trades=backtest["n_trades"],
                min_required=MIN_BACKTEST_TRADES,
            )

        # LogReg quality metrics are advisory; only technical failures block use.
        should_activate = True
        if not passed_quality_gate:
            logger.warning(
                "model_quality_gate_failed",
                asset=asset,
                reasons=gate_reasons,
                val_auc=val_acc,
                baseline_auc=baseline_acc,
                ece=ece,
                action="Diagnostic warning only. Model remains eligible for PAPER activation."
            )
        else:
            logger.info("model_quality_gate_passed", asset=asset, val_auc=val_acc, baseline_auc=baseline_acc)

        # Получаем предыдущую активную модель для сравнения AUC
        prev_auc_res = await self.db.execute(
            select(ModelRegistry.accuracy)
            .where(ModelRegistry.asset == asset, ModelRegistry.is_active == True)
            .order_by(ModelRegistry.version.desc())
            .limit(1)
        )
        prev_auc = prev_auc_res.scalar_one_or_none()

        # ШАГ 1.1 FIX: проверка min_auc ДО каких-либо изменений в БД
        from polyflip.services.settings_service import get_float

        min_auc_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == f"MIN_AUC_{asset}")
        )).scalar_one_or_none()
        min_auc = float(min_auc_row.value) if min_auc_row else await get_float(self.db, "LR_MIN_AUC_FOR_DEPLOY")

        if val_acc < min_auc:
            logger.warning(
                "model_quality_below_threshold",
                asset=asset,
                val_auc=round(val_acc, 4),
                min_auc_required=min_auc,
            )
            self.status_messages[asset] = (
                f"Предупреждение: AUC {val_acc:.3f} < min_auc {min_auc:.2f}; "
                f"LogReg активирована для непрерывной PAPER-оценки"
            )

        # Если модель прошла проверку, деактивируем старые записи
        if should_activate:
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

        # Сохраняем калиброванный порог в RuntimeSettings только если модель активируется или если нет настроек
        threshold_key = f"AUTO_FLIP_THRESHOLD_{asset}"
        if save_settings:
            existing = await self.db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == threshold_key)
            )
            existing_row = existing.scalar_one_or_none()
            if should_activate or not existing_row:
                if existing_row:
                    existing_row.value = str(round(optimal_threshold, 4))
                else:
                    self.db.add(RuntimeSettings(
                        key=threshold_key,
                        value=str(round(optimal_threshold, 4)),
                        updated_at=datetime.now(timezone.utc),
                        updated_by="train_job"
                    ))

        # 7. Сохраняем новую модель
        new_model_record = ModelRegistry(
            asset=asset,
            version=next_version,
            model_type="logreg",
            model_blob=model_bytes,
            accuracy=val_acc,
            baseline=baseline_acc,
            features=",".join(active_features),
            ece=ece,
            is_active=should_activate,
            decision_threshold=optimal_threshold,
            quality_gate_passed=passed_quality_gate,
            quality_gate_reasons={
                "reasons": gate_reasons, "auc": val_acc, "ece": ece,
                "backtest": backtest,
            },
            training_params={
                "quality_gate_mode": "ADVISORY",
                "backtest_strategy_branch": backtest["strategy_branch"],
                "target_source": "POLYMARKET_FLIP_VS_FINAL_OUTCOME",
                "validation_scheme": backtest["validation_scheme"],
                "validation_folds": backtest["validation_folds"],
                "feature_set_version": FEATURE_EXPERIMENT_LABELS.get(
                    experiment_variant,
                    SEQUENCE_FEATURE_SET_VERSION if sequence_enabled else "baseline-v1",
                ),
                "experiment_variant": experiment_variant,
                "sequence_coverage": round(sequence_coverage, 6),
                "sequence_source": "CLOSED_UNDERLYING_15M",
                "model_config": backtest["model_config"],
                "brier_score": backtest["brier_score"],
                "log_loss": backtest["log_loss"],
                "oot_markets": backtest["oot_markets"],
            },
            activation_source="TRAINER",
            quality_override=not passed_quality_gate,
            activated_at=datetime.now(timezone.utc),
            activated_by="trainer",
            interval="15m",
            dataset_fingerprint=dataset_fingerprint,
            training_window_start=df["_decision_at"].min().to_pydatetime(),
            training_window_end=df["_decision_at"].max().to_pydatetime(),
            trained_at=datetime.now(timezone.utc),
            backtest_pnl=backtest["total_pnl"],
            backtest_trades=backtest["n_trades"],
            backtest_wr=backtest["win_rate"],
        )

        self.db.add(new_model_record)
        await self.db.commit()

        logger.info("model_saved_to_db", asset=asset, version=next_version, threshold=optimal_threshold)
        
        diff_str = ""
        if prev_auc is not None:
            diff = val_acc - prev_auc
            if diff > 0.0001:
                diff_str = f" (+{diff:.4f} 🟢 лучше)"
            elif diff < -0.0001:
                diff_str = f" ({diff:.4f} 🔴 хуже)"
            else:
                diff_str = " (= без изм.)"

        self.status_messages[asset] = f"Успешно: версия {next_version} (AUC {val_acc:.4f}{diff_str})"

        # --- Шаг 3: Price-Phase Split ---
        from polyflip.constants import PRICE_PHASE_BOUNDARIES, CV_N_SPLITS
        from polyflip.services.settings_service import get_int
        min_samples = await get_int(self.db, "MIN_SAMPLES_FOR_PHASE_MODEL")
        
        assert "price_deviation" in df.columns, (
            "price_deviation must be computed before phase split. "
            "Call add_derived_features(df) first."
        )
        
        phase_results = {}
        for phase_name, (lo, hi) in PRICE_PHASE_BOUNDARIES.items():
            df_phase = df[
                (df["price_deviation"] >= lo) & (df["price_deviation"] < hi)
            ].copy()

            n_phase = len(df_phase)
            logger.info("price_phase_split_stats", asset=asset, phase=phase_name, n=n_phase,
                        target_mean=round(df_phase["target"].mean(), 3) if n_phase > 0 else None)

            if n_phase < min_samples:
                logger.warning("price_phase_model_skipped", asset=asset, phase=phase_name,
                               n=n_phase, required=min_samples)
                phase_results[phase_name] = f"skipped ({n_phase} samples)"
                continue

            if len(df_phase["target"].unique()) < 2:
                logger.warning("price_phase_one_class", asset=asset, phase=phase_name)
                phase_results[phase_name] = "skipped (one class)"
                continue
                
            n_unique_markets = df_phase["market_id"].nunique()
            if n_unique_markets < CV_N_SPLITS:
                logger.warning(
                    "price_phase_not_enough_groups",
                    asset=asset, phase=phase_name,
                    n_markets=n_unique_markets, required=CV_N_SPLITS,
                )
                phase_results[phase_name] = f"skipped ({n_unique_markets} markets < {CV_N_SPLITS} folds)"
                continue

            phase_asset = f"{asset}_{phase_name}"
            X_phase = df_phase[active_features]
            y_phase = df_phase["target"]
            grp_phase = df_phase["market_id"]

            phase_weights = (
                _compute_sample_weights(
                    df_phase["time_left_min"].values,
                    mode=weight_mode,
                    tau=weight_tau,
                )
                if weight_mode != "uniform"
                else None
            )

            try:
                sem_phase = await _get_training_semaphore(self.db)
                async with sem_phase:
                    fit_res_phase = await asyncio.to_thread(
                        _fit_and_serialize,
                        X_phase,
                        y_phase,
                        grp_phase,
                        mid_prices=df_phase["mid_price"],
                        sample_weight=phase_weights,
                        lr_coef_threshold=lr_coef_threshold,
                        lr_min_features=lr_min_features,
                        timestamps=df_phase["_decision_at"],
                        min_precision=min_precision,
                        max_suspicious=max_suspicious,
                        fee_per_trade=fee_per_trade,
                    )
            except Exception as e:
                logger.error("price_phase_fit_failed", asset=asset, phase=phase_name, error=str(e))
                phase_results[phase_name] = f"failed: {e}"
                continue
                
            if not fit_res_phase:
                continue

            model_bytes_p, val_acc_p, baseline_acc_p, threshold_p, ece_p, backtest_p = fit_res_phase

            phase_gate_reasons = []
            if val_acc_p < min_auc:
                phase_gate_reasons.append(f"AUC {val_acc_p:.4f} below {min_auc:.4f}")
            if ece_p > 0.15:
                phase_gate_reasons.append(f"ECE {ece_p:.4f} above 0.15")
            if threshold_p < MODEL_THRESHOLD_MIN or threshold_p > MODEL_THRESHOLD_MAX:
                phase_gate_reasons.append(
                    f"Threshold {threshold_p:.4f} outside advisory bounds"
                )
            if backtest_p["n_trades"] >= MIN_BACKTEST_TRADES and backtest_p["total_pnl"] < MIN_BACKTEST_PNL:
                phase_gate_reasons.append(
                    f"Backtest PnL {backtest_p['total_pnl']:.4f} below {MIN_BACKTEST_PNL:.4f}"
                )

            # Деактивируем старые
            await self.db.execute(
                update(ModelRegistry).where(ModelRegistry.asset == phase_asset).values(is_active=False)
            )
            last_v_p = (await self.db.execute(
                select(ModelRegistry.version).where(ModelRegistry.asset == phase_asset)
                .order_by(ModelRegistry.version.desc()).limit(1)
            )).scalar_one_or_none()

            if save_settings:
                # Сохраняем порог
                thr_key = f"AUTO_FLIP_THRESHOLD_{phase_asset}"
                existing_thr = (await self.db.execute(
                    select(RuntimeSettings).where(RuntimeSettings.key == thr_key)
                )).scalar_one_or_none()
                if existing_thr:
                    existing_thr.value = str(round(threshold_p, 4))
                else:
                    self.db.add(RuntimeSettings(
                        key=thr_key, value=str(round(threshold_p, 4)),
                        updated_at=datetime.now(timezone.utc), updated_by="train_job_phase"
                    ))

            self.db.add(ModelRegistry(
                asset=phase_asset, version=(last_v_p or 0) + 1,
                model_blob=model_bytes_p, accuracy=val_acc_p,
                baseline=baseline_acc_p, features=",".join(active_features),
                ece=ece_p, is_active=True, interval="15m",
                trained_at=datetime.now(timezone.utc),
                backtest_pnl=backtest_p["total_pnl"],
                decision_threshold=threshold_p,
                quality_gate_passed=not phase_gate_reasons,
                quality_gate_reasons={
                    "reasons": phase_gate_reasons, "auc": val_acc_p,
                    "ece": ece_p, "backtest": backtest_p,
                },
                activation_source="TRAINER",
                training_params={
                    "quality_gate_mode": "ADVISORY",
                    "backtest_strategy_branch": backtest_p["strategy_branch"],
                    "target_source": "POLYMARKET_FLIP_VS_FINAL_OUTCOME",
                    "validation_scheme": backtest_p["validation_scheme"],
                    "validation_folds": backtest_p["validation_folds"],
                    "feature_set_version": FEATURE_EXPERIMENT_LABELS.get(
                        experiment_variant,
                        SEQUENCE_FEATURE_SET_VERSION if sequence_enabled else "baseline-v1",
                    ),
                    "experiment_variant": experiment_variant,
                    "sequence_coverage": round(sequence_coverage, 6),
                    "model_config": backtest_p["model_config"],
                    "oot_markets": backtest_p["oot_markets"],
                },
                quality_override=bool(phase_gate_reasons),
                activated_at=datetime.now(timezone.utc),
                activated_by="trainer",
                backtest_trades=backtest_p["n_trades"],
                backtest_wr=backtest_p["win_rate"],
            ))
            phase_state = "warning" if phase_gate_reasons else "ok"
            phase_results[phase_name] = (
                f"{phase_state} (AUC {val_acc_p:.3f}, n={n_phase})"
            )

        await self.db.commit()
        logger.info("price_phase_models_complete", asset=asset, results=phase_results)
        if phase_results:
            phase_summary = ", ".join(f"{k}: {v}" for k, v in phase_results.items())
            self.status_messages[asset] = f"{self.status_messages.get(asset, '')} | Фазы: [{phase_summary}]"

        return True

    async def train(self, asset: str, save_settings: bool = True) -> bool:
        return await self.train_model(asset, save_settings=save_settings)
