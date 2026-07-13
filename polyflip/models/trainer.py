import pickle
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve

from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.config import settings
from polyflip.constants import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    MIN_PRECISION_FOR_THRESHOLD,
    MAX_SUSPICIOUS_THRESHOLD
)

logger = structlog.get_logger(__name__)

from polyflip.models.feature_lags import add_lag_features, LAG_FEATURE_NAMES

DERIVED_FEATURES = [
    "price_deviation",
    "deviation_x_time",
    "price_deviation_sq",
    "spread_pct",
    "log_time_left",
    "day_of_week",
    "price_distance_from_max",
    "time_phase",
    *LAG_FEATURE_NAMES,
]

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_deviation"]     = (df["mid_price"] - 0.5).abs()
    df["deviation_x_time"]    = df["price_deviation"] * df["time_left_min"]
    df["price_deviation_sq"]  = df["price_deviation"] ** 2
    df["spread_pct"]          = (df["spread"] / (df["mid_price"] + 1e-6)).clip(upper=10.0)
    df["log_time_left"]       = np.log1p(df["time_left_min"])

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

    if "market_id" in df.columns and "time_left_min" in df.columns:
        df["time_phase"] = (df["time_left_min"] / (df.groupby("market_id")["time_left_min"].transform("max") + 1e-6)).clip(0, 1)
    else:
        df["time_phase"] = 1.0

    return df

def _fit_and_serialize(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Синхронная CPU-bound функция для кросс-валидации, обучения и сериализации модели."""
    # --- Grid search по C ---
    C_GRID = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    gkf_search = GroupKFold(n_splits=CV_N_SPLITS)
    c_results = {}
    
    for c_val in C_GRID:
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", C=c_val,
                random_state=CV_RANDOM_STATE, max_iter=1000,
            )),
        ])
        probe_aucs = []
        for tr_idx, vl_idx in gkf_search.split(X, y, groups=groups):
            if len(np.unique(y.iloc[tr_idx])) < 2 or len(np.unique(y.iloc[vl_idx])) < 2:
                continue
            m = clone(probe)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = m.predict_proba(X.iloc[vl_idx])[:, 1]
            probe_aucs.append(roc_auc_score(y.iloc[vl_idx], proba))
        if probe_aucs:
            c_results[c_val] = round(float(np.mean(probe_aucs)), 4)
    
    best_C = max(c_results, key=c_results.get) if c_results else 5.0
    logger.info("c_grid_search_results", c_grid=c_results, best_C=best_C)

    # 3. Обучаем модель с кросс-валидацией
    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", C=best_C, random_state=CV_RANDOM_STATE, max_iter=1000))
    ])
    
    from sklearn.calibration import CalibratedClassifierCV
    
    aucs = []
    oof_scores = np.zeros(len(y))
    for train_index, val_index in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        
        fold_base = clone(base_model)
        fold_base.fit(X_train, y_train)
        
        from sklearn.frozen import FrozenEstimator
        fold_calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(fold_base),
            method="sigmoid",
            cv=[([], np.arange(len(y_val)))]
        )
        fold_calibrated.fit(X_val, y_val)
        
        y_proba = fold_calibrated.predict_proba(X_val)[:, 1]
        oof_scores[val_index] = y_proba
        aucs.append(roc_auc_score(y_val, y_proba))
        
    val_acc = float(np.mean(aucs)) if aucs else 0.5
    
    # Baseline ROC-AUC/Accuracy (доля мажоритарного класса)
    baseline_acc = float(max(y.mean(), 1.0 - y.mean()))
    
    # ECE Diagnostic
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y, oof_scores, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    logger.info("calibration_check", ece=round(ece, 4))
    
    # Обучаем финальную модель на всех данных (с holdout для честной калибровки)
    from sklearn.model_selection import train_test_split
    min_class_count = int(y.value_counts().min())
    use_stratify = y if min_class_count >= 10 else None
    
    X_train_cal, X_cal, y_train_cal, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=use_stratify
    )
    
    if len(np.unique(y_train_cal)) < 2 or len(np.unique(y_cal)) < 2:
        # Fallback to uncalibrated model on entire dataset if split is invalid
        final_model = clone(base_model)
        try:
            final_model.fit(X, y)
        except Exception:
            return None # Impossible to fit
    else:
        final_base = clone(base_model)
        final_base.fit(X_train_cal, y_train_cal)
        
        from sklearn.frozen import FrozenEstimator
        final_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(final_base),
            method="sigmoid",
            cv=[([], np.arange(len(y_cal)))]
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

    from polyflip.constants import LR_COEF_THRESHOLD, LR_MIN_FEATURES
    weak_features = [f for f, v in abs_coefs if v < LR_COEF_THRESHOLD]
    if len(X.columns) - len(weak_features) >= LR_MIN_FEATURES and weak_features:
        logger.warning(
            "weak_features_detected",
            count=len(weak_features),
            features=weak_features,
            threshold=LR_COEF_THRESHOLD,
            suggestion="Consider removing from ACTIVE_FEATURES via dashboard",
        )
    
    # Калибровка порога с использованием Out-Of-Fold предсказаний (исключаем Data Leakage)
    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y, oof_scores)

    # Найти порог с лучшим F1 среди тех где precision >= MIN_PRECISION_FOR_THRESHOLD
    valid_mask = precision_arr[:-1] >= MIN_PRECISION_FOR_THRESHOLD
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

    # Проверка: если leakage есть — порог будет подозрительно высоким
    if optimal_threshold >= MAX_SUSPICIOUS_THRESHOLD:
        raise ValueError(
            f"Подозрительный порог {optimal_threshold:.3f} >= {MAX_SUSPICIOUS_THRESHOLD:.2f} — "
            "вероятно data leakage при калибровке. Проверь OOF-скоры."
        )

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
        min_precision_used=MIN_PRECISION_FOR_THRESHOLD,
        n_samples=len(y),
        fold_aucs=[round(a, 4) for a in aucs],
    )

    # Сериализуем модель (Pipeline сохраняет скейлер внутри)
    model_bytes = pickle.dumps(final_model)
    return model_bytes, val_acc, baseline_acc, optimal_threshold, ece

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
        
        # 1. Сначала проверяем количество доступных сэмплов через быстрый COUNT(*)
        min_time_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_MIN_TIME_LEFT_SEC")
        max_time_stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_MAX_TIME_LEFT_SEC")

        min_time_row = (await self.db.execute(min_time_stmt)).scalar_one_or_none()
        max_time_row = (await self.db.execute(max_time_stmt)).scalar_one_or_none()

        min_time_sec = int(min_time_row.value) if min_time_row else settings.TRADE_MIN_TIME_LEFT_SEC
        max_time_sec = int(max_time_row.value) if max_time_row else settings.TRADE_MAX_TIME_LEFT_SEC

        min_time_min = min_time_sec / 60.0
        max_time_min = max_time_sec / 60.0

        count_stmt = select(func.count(MarketSnapshot.id)).where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.final_outcome != "PENDING",
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

        # Получаем обучающую выборку (исключаем PENDING), так как данных достаточно
        stmt = select(MarketSnapshot).where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.final_outcome != "PENDING",
            MarketSnapshot.time_left_min >= min_time_min,
            MarketSnapshot.time_left_min <= max_time_min
        )
        result = await self.db.execute(stmt)
        snapshots = result.scalars().all()

        # 2. Формируем DataFrame
        data = []
        for s in snapshots:
            data.append({
                "market_id": s.market_id,
                "recorded_at": s.recorded_at,
                "time_left_min": s.time_left_min,
                "mid_price": s.mid_price,
                "spread": s.spread,
                "price_velocity": s.price_velocity,
                "volume_5min": s.volume_5min,
                "hour_of_day": s.hour_of_day,
                "day_of_week": s.recorded_at.weekday(),
                "target": 1 if s.flip_vs_final else 0
            })
            
        df = pd.DataFrame(data)
        
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
        df.drop(columns=["recorded_at"], errors="ignore", inplace=True)

        # Автоматически расширяем active_features производными признаками,
        # если их базовые источники (mid_price, spread, time_left_min) присутствуют
        base_for_derived = {"mid_price", "spread", "time_left_min"}
        if base_for_derived.issubset(set(active_features)):
            for feat in DERIVED_FEATURES:
                if feat not in active_features:
                    active_features.append(feat)
            logger.info("derived_features_added", features=DERIVED_FEATURES, asset=asset)
            
            # Синхронизируем расширенный список с БД RuntimeSettings
            derived_setting = await self.db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == "ACTIVE_FEATURES")
            )
            derived_row = derived_setting.scalar_one_or_none()
            if derived_row:
                new_value = ",".join(active_features)
                if derived_row.value != new_value:
                    derived_row.value = new_value
        
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
        groups = df["market_id"]

        # Выполняем CPU-bound обучение в отдельном потоке (BUG-A2 FIX)
        fit_res = await asyncio.to_thread(_fit_and_serialize, X, y, groups)
        model_bytes, val_acc, baseline_acc, optimal_threshold, ece = fit_res

        logger.info("model_trained", asset=asset, samples=len(df), val_auc=val_acc, baseline_auc=baseline_acc, ece=ece)

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

        # Сохраняем калиброванный порог в RuntimeSettings
        threshold_key = f"AUTO_FLIP_THRESHOLD_{asset}"
        existing = await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == threshold_key)
        )
        existing_row = existing.scalar_one_or_none()
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
            model_blob=model_bytes,
            accuracy=val_acc,
            baseline=baseline_acc,
            features=",".join(active_features),
            ece=ece,
            is_active=True,
            interval="15m",
            trained_at=datetime.now(timezone.utc)
        )

        from polyflip.constants import LR_MIN_AUC_FOR_DEPLOY

        min_auc_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == f"MIN_AUC_{asset}")
        )).scalar_one_or_none()
        min_auc = float(min_auc_row.value) if min_auc_row else LR_MIN_AUC_FOR_DEPLOY

        if val_acc < min_auc:
            logger.warning(
                "model_quality_below_threshold",
                asset=asset,
                val_auc=round(val_acc, 4),
                min_auc_required=min_auc,
            )
            self.status_messages[asset] = (
                f"Пропущено: AUC {val_acc:.3f} < min_auc {min_auc:.2f} — "
                f"модель не задеплоена, используется предыдущая версия"
            )
            return False
            
        self.db.add(new_model_record)
        await self.db.commit()

        logger.info("model_saved_to_db", asset=asset, version=next_version, threshold=optimal_threshold)
        self.status_messages[asset] = f"Успешно: версия {next_version} (AUC {val_acc:.2f})"
        return True
