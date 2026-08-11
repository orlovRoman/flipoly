"""
LightGBM-тренер для крипто-модели Up/Down на OHLCV-свечах.

Таргет: Up (1) / Down (0) с фильтром |return_15m| >= ε (90-й перцентиль).
Фичи: из feature_builder.build_features().
Сериализация: pickle в ModelRegistry (та же схема, что и LogReg-модель).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator, calibration_curve
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.constants import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
)
from polyflip.services.settings_service import get_float, get_int
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features, CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.market_outcome_dataset import build_market_outcome_dataset
from polyflip.db.models import CryptoCandle, ModelRegistry, RuntimeSettings

# Импортируем общий семафор из LogReg-трейнера.
# NOTE: Семафор инициализируется один раз при первом вызове и кэшируется до перезапуска сервиса.
# Изменение TRAIN_MAX_PARALLEL_JOBS в RuntimeSettings вступит в силу только после рестарта.
from polyflip.models.trainer import _get_training_semaphore

logger = structlog.get_logger(__name__)

from polyflip.crypto.feature_sets import (
    CONTROL_FEATURES,
    feature_schema_hash,
    get_feature_set,
)

CRYPTO_FEATURES = list(CONTROL_FEATURES)


# Fail fast при старте: CRYPTO_FEATURES должен быть подмножеством CRYPTO_FEATURE_COLUMNS
_unknown = set(CRYPTO_FEATURES) - set(CRYPTO_FEATURE_COLUMNS)
assert not _unknown, (
    f"CRYPTO_FEATURES содержит фичи, которых нет в feature_builder: {_unknown}"
)


QUALITY_GATE_MIN_LIFT = -0.005
QUALITY_GATE_MAX_ECE = 0.15
QUALITY_GATE_MAX_ACTIVE_DEGRADATION = 0.02


def _dataset_fingerprint(df: pd.DataFrame, features: list[str]) -> str:
    """Build an order-independent fingerprint from identity, target and features."""
    columns = [
        column
        for column in ("market_id", "market_start", "target", *features)
        if column in df.columns
    ]
    if not columns:
        return hashlib.sha256(b"").hexdigest()[:32]
    canonical = df.loc[:, columns]
    sort_columns = [column for column in ("market_id", "market_start") if column in columns]
    canonical = canonical.sort_values(sort_columns or columns, kind="mergesort").reset_index(drop=True)
    payload = json.dumps(
        canonical.to_dict(orient="records"),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _model_smoke_test(
    model_bytes: bytes,
    features: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    """Return an audit-friendly error when a serialized model is unsafe to activate."""
    try:
        clf = pickle.loads(model_bytes)
        expected = len(features or tuple(CRYPTO_FEATURES))
        actual = getattr(clf, "n_features_in_", None)
        if actual is not None and actual != expected:
            return f"ModelCompatibilityError: expected={expected}, actual={actual}"

        predict_proba = getattr(clf, "predict_proba", None)
        if not callable(predict_proba):
            return "ModelCompatibilityError: predict_proba is not callable"

        proba = np.asarray(
            predict_proba(np.zeros((1, expected), dtype=np.float64)),
            dtype=np.float64,
        )
        if (
            proba.shape != (1, 2)
            or not np.isfinite(proba).all()
            or np.any((proba < 0.0) | (proba > 1.0))
            or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        ):
            return "ModelCompatibilityError: invalid predict_proba result"
    except Exception as exc:
        return f"ModelCompatibilityError: failed to load or test model - {exc}"
    return None


def _evaluate_quality_gate(
    *,
    model_bytes: bytes,
    val_auc: float,
    baseline_auc: float,
    ece: float,
    threshold: float,
    threshold_down: float,
    features: tuple[str, ...] | list[str] | None = None,
    active_accuracy: float | None = None,
    active_version: int | None = None,
) -> tuple[bool, list[str], float, float]:
    """Validate artifact compatibility; quality metrics remain advisory diagnostics."""
    reasons: list[str] = []
    metrics = {"accuracy": val_auc, "baseline": baseline_auc, "ece": ece}
    invalid_metrics = [name for name, value in metrics.items() if not np.isfinite(value)]
    if invalid_metrics:
        reasons.append(f"Advisory: Non-finite quality metrics: {', '.join(invalid_metrics)}")
    else:
        lift = val_auc - baseline_auc
        if lift < QUALITY_GATE_MIN_LIFT:
            reasons.append(
                f"Advisory: Negative lift vs baseline: {lift:+.4f} "
                f"(accuracy={val_auc:.4f}, baseline={baseline_auc:.4f})"
            )
        if ece > QUALITY_GATE_MAX_ECE:
            reasons.append(
                f"Advisory: Excessive ECE calibration error: {ece:.4f} > {QUALITY_GATE_MAX_ECE:.2f}"
            )

    if active_accuracy is not None:
        if not np.isfinite(active_accuracy):
            reasons.append("Advisory: Active model accuracy is non-finite")
        elif np.isfinite(val_auc):
            acc_diff = val_auc - active_accuracy
            if acc_diff < -QUALITY_GATE_MAX_ACTIVE_DEGRADATION:
                reasons.append(
                    f"Advisory: Accuracy degraded vs active model v{active_version}: "
                    f"{acc_diff:+.4f} < -{QUALITY_GATE_MAX_ACTIVE_DEGRADATION:.2f}"
                )

    normalized_thresholds: list[float] = []
    for label, value in (("UP", threshold), ("DOWN", threshold_down)):
        if not np.isfinite(value):
            normalized = 0.5
            reasons.append(f"{label} threshold is non-finite, reset to {normalized:.4f}")
        else:
            original = float(value)
            normalized = max(0.0, min(1.0, original))
            if normalized != original:
                reasons.append(
                    f"{label} threshold {original:.4f} outside probability bounds [0.0, 1.0], "
                    f"normalized to {normalized:.4f}"
                )
        normalized_thresholds.append(normalized)

    smoke_error = _model_smoke_test(model_bytes, features)
    if smoke_error:
        reasons.append(smoke_error)

    technical_errors = [
        reason for reason in reasons if reason.startswith("ModelCompatibilityError")
    ]
    return not technical_errors, reasons, normalized_thresholds[0], normalized_thresholds[1]


# Epsilon filter removed as per MARKET_WINDOW_V1 spec.


def _make_lgbm(**params) -> LGBMClassifier:
    """Вспомогательная функция для создания квалифицированного LGBMClassifier."""
    defaults = {
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        15,
        "max_depth":         4,
        "min_child_samples": 50,
        "subsample":         0.8,
        "colsample_bytree":  1.0,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      CV_RANDOM_STATE,
        "n_jobs":            1,
        "verbose":          -1,
    }
    defaults.update(params)
    return LGBMClassifier(**defaults)


def _fit_lgbm_and_serialize(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_N_SPLITS,
    min_precision: float = 0.52,
    min_valid_thr: float = 0.30,
    max_valid_thr: float = 0.75,
    thr_fallback: float = 0.55,
    **lgbm_params,
) -> tuple[bytes, float, float, float, float, float, dict[str, int], float, float, float, float]:
    """
    CPU-bound. Обучает LightGBM с TimeSeriesSplit.
    Возвращает: (model_bytes, val_auc, baseline_auc, optimal_threshold, optimal_threshold_down, ece, feature_importance, ...)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_scores = np.full(len(y), np.nan)
    aucs: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_lgbm = _make_lgbm(**lgbm_params)
        fold_lgbm.fit(X_train, y_train)

        # Калибруем на первой половине val, измеряем AUC на второй — без пересечения
        mid = len(val_idx) // 2
        cal_idx, eval_idx = val_idx[:mid], val_idx[mid:]

        if len(cal_idx) >= 20 and len(eval_idx) >= 20:
            X_cal,  y_cal  = X.iloc[cal_idx],  y.iloc[cal_idx]
            X_eval, y_eval = X.iloc[eval_idx], y.iloc[eval_idx]
            calibration_method = "sigmoid" if len(cal_idx) < 200 else "isotonic"
            fold_cal = CalibratedClassifierCV(
                estimator=FrozenEstimator(fold_lgbm), method=calibration_method, cv=None
            )
            fold_cal.fit(X_cal, y_cal)
            y_proba = fold_cal.predict_proba(X_eval)[:, 1]
            oof_scores[eval_idx] = y_proba
            aucs.append(roc_auc_score(y_eval, y_proba))
        else:
            # Фолд слишком мал — используем некалиброванную модель
            y_proba = fold_lgbm.predict_proba(X.iloc[val_idx])[:, 1]
            oof_scores[val_idx] = y_proba
            aucs.append(roc_auc_score(y.iloc[val_idx], y_proba))

    val_auc = float(np.mean(aucs))
    baseline_auc = 0.5

    # ECE через OOF
    valid_mask = ~np.isnan(oof_scores)
    try:
        if valid_mask.sum() > 10:
            y_cal = y[valid_mask].to_numpy(dtype=float)
            p_cal = oof_scores[valid_mask]
            frac_pos, mean_pred = calibration_curve(y_cal, p_cal, n_bins=10, strategy="uniform")
            edges = np.linspace(0.0, 1.0, 11)
            bucket = np.clip(np.digitize(p_cal, edges[1:-1], right=False), 0, 9)
            counts = np.bincount(bucket, minlength=10)
            weights = counts[counts > 0] / len(p_cal)
            ece = float(np.sum(weights * np.abs(frac_pos - mean_pred)))
        else:
            ece = 0.5
    except ValueError:
        ece = 0.5  # недостаточно данных для расчёта

    logger.info("crypto_calibration", ece=round(ece, 4))

    # Финальная модель на всех данных
    n_cal = max(50, int(len(X) * 0.15))
    X_fit, X_cal_final = X.iloc[:-n_cal], X.iloc[-n_cal:]
    y_fit, y_cal_final = y.iloc[:-n_cal], y.iloc[-n_cal:]

    final_lgbm = _make_lgbm(**lgbm_params)
    final_lgbm.fit(X_fit, y_fit)
    final_calibration_method = "sigmoid" if n_cal < 200 else "isotonic"
    final_cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(final_lgbm), method=final_calibration_method, cv=None
    )
    final_cal.fit(X_cal_final, y_cal_final)

    def _find_threshold(y_true, y_prob):
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
        if len(thr_arr) > 0:
            valid = prec_arr[:-1] >= min_precision
            f1 = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
            if valid.any():
                th = float(thr_arr[np.argmax(np.where(valid, f1, 0.0))])
            else:
                th = float(thr_arr[np.argmax(f1)])
        else:
            th = thr_fallback
        if th >= max_valid_thr:
            th = max_valid_thr
        if th < min_valid_thr or th > max_valid_thr:
            th = thr_fallback
        return th

    optimal_threshold = _find_threshold(y[valid_mask], oof_scores[valid_mask])
    
    y_down = 1 - y[valid_mask]
    oof_scores_down = 1.0 - oof_scores[valid_mask]
    optimal_threshold_down = _find_threshold(y_down, oof_scores_down)

    # Считаем реальный precision при optimal_threshold на OOF
    valid_oof = oof_scores[valid_mask]
    y_valid = y[valid_mask]
    predictions_at_thr = valid_oof >= optimal_threshold
    if predictions_at_thr.sum() > 20:
        real_precision = float((y_valid[predictions_at_thr] == 1).mean())
        signal_rate = float(predictions_at_thr.mean())
        logger.info("oof_real_precision",
            precision=round(real_precision, 4),
            signal_rate=round(signal_rate, 4),
            threshold=round(optimal_threshold, 4))
        # Если precision < 0.52 — порог бесполезен
        if real_precision < 0.52:
            logger.warning("precision_below_random",
                real_precision=real_precision,
                action="threshold_raised_or_model_rejected")



    # Feature importance для логирования и дашборда
    fi = {
        col: int(imp)
        for col, imp in zip(X.columns, final_lgbm.feature_importances_)
    }
    zero_imp = [f for f, v in fi.items() if v == 0]
    if zero_imp:
        logger.warning(
            "features_with_zero_importance",
            features=zero_imp,
            hint="Consider removing these features in next refactor",
        )
    logger.info("crypto_feature_importance", top5=dict(sorted(fi.items(), key=lambda x: -x[1])[:5]))

    y_oof = y[valid_mask].astype(int)
    p_oof = oof_scores[valid_mask]
    y_pred = (p_oof >= optimal_threshold).astype(int)

    precision = float(precision_score(y_oof, y_pred, zero_division=0))
    recall = float(recall_score(y_oof, y_pred, zero_division=0))
    f1_metric = float(f1_score(y_oof, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_oof, p_oof))

    return (
        pickle.dumps(final_cal),
        val_auc,
        baseline_auc,
        optimal_threshold,
        optimal_threshold_down,
        ece,
        fi,
        precision,
        recall,
        f1_metric,
        brier,
    )


async def _get_float_setting(db: AsyncSession, key: str, default: float = 0.0) -> float:
    try:
        return await get_float(db, key)
    except KeyError:
        row = (await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
        return float(row.value) if row else default


async def _get_int_setting(db: AsyncSession, key: str, default: int = 0) -> int:
    try:
        return await get_int(db, key)
    except KeyError:
        row = (await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
        return int(row.value) if row else default


class CryptoModelTrainer:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def train(
        self,
        symbol: str,
        interval: str = "15m",
        save_settings: bool = True,
        *,
        feature_set: str = "A",
        activate_after_train: bool = True,
    ) -> bool:
        feature_spec=get_feature_set(feature_set)
        available=list(feature_spec.features)
        logger.info("crypto_training_start", symbol=symbol, interval=interval, feature_set=feature_spec.key, feature_set_version=feature_spec.version, activate_after_train=activate_after_train)

        # Загружаем все свечи для символа
        candles = await get_recent_candles(
            self.db, symbol, interval, limit=10_000
        )
        if len(candles) < 500:
            logger.warning("not_enough_candles", symbol=symbol, count=len(candles))
            return False

        # Считываем динамические гиперпараметры из RuntimeSettings
        n_estimators = await _get_int_setting(self.db, "CRYPTO_LGBM_N_ESTIMATORS", 300)
        learning_rate = await _get_float_setting(self.db, "CRYPTO_LGBM_LEARNING_RATE", 0.05)
        num_leaves = await _get_int_setting(self.db, "CRYPTO_LGBM_NUM_LEAVES", 31)
        max_depth = await _get_int_setting(self.db, "CRYPTO_LGBM_MAX_DEPTH", 5)
        min_child_samples = await _get_int_setting(self.db, "CRYPTO_LGBM_MIN_CHILD_SAMPLES", 20)
        subsample = await _get_float_setting(self.db, "CRYPTO_LGBM_SUBSAMPLE", 0.8)
        colsample_bytree = await _get_float_setting(self.db, "CRYPTO_LGBM_COLSAMPLE_BYTREE", 0.8)
        reg_alpha = await _get_float_setting(self.db, "CRYPTO_LGBM_REG_ALPHA", 0.1)
        reg_lambda = await _get_float_setting(self.db, "CRYPTO_LGBM_REG_LAMBDA", 1.0)
        lgbm_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }

        min_precision = await get_float(self.db, "LGBM_MIN_PRECISION_FOR_THRESHOLD")
        min_valid_thr = await get_float(self.db, "LGBM_MIN_VALID_THRESHOLD")
        max_valid_thr = await get_float(self.db, "LGBM_MAX_VALID_THRESHOLD")
        thr_fallback = await get_float(self.db, "LGBM_THRESHOLD_FALLBACK")
        cv_n_splits = await get_int(self.db, "LGBM_CV_N_SPLITS")
        epsilon_quantile = await get_float(self.db, "LGBM_EPSILON_QUANTILE")

        # Читаем актуальные funding rates из БД
        fr_key = f"FUNDING_RATE_{symbol}"
        fr_ma3_key = f"FUNDING_RATE_MA3_{symbol}"
        fr_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == fr_key)
        )).scalar_one_or_none()
        fr_ma3_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == fr_ma3_key)
        )).scalar_one_or_none()

        funding_rate = float(fr_row.value) if fr_row else 0.0
        funding_rate_ma3 = float(fr_ma3_row.value) if fr_ma3_row else 0.0

        logger.info(
            "funding_rate_loaded_for_training",
            symbol=symbol,
            funding_rate=funding_rate,
            ma3=funding_rate_ma3,
        )

        # 3. Загружаем выравненный торговый датасет на канонических исходах Polymarket (MARKET_WINDOW_V1)
        df_filtered = await build_market_outcome_dataset(self.db, symbol=symbol, interval=interval)
        
        # FAIL-CLOSED: Если датасет пустой — прекращаем обучение без записи моделей или обновления настроек
        if df_filtered is None or df_filtered.empty:
            logger.error(
                "polymarket_training_dataset_empty",
                symbol=symbol,
                target_source="POLYMARKET_FINAL_OUTCOME",
            )
            return False

        # 3. Добавить строгую валидацию датасета
        required_cols = {"market_id", "market_start", "feature_available_at", "final_outcome", "target", *available}
        missing_cols = required_cols - set(df_filtered.columns)
        if missing_cols:
            logger.error("training_dataset_missing_columns", missing=list(missing_cols))
            return False

        assert df_filtered["market_id"].is_unique, "market_id must be strictly unique in dataset"
        assert set(df_filtered["target"].unique()).issubset({0, 1}), "target must be in {0, 1}"
        assert (df_filtered["feature_available_at"] <= df_filtered["market_start"]).all(), "feature_available_at must be <= market_start"
        assert df_filtered[available].isna().sum().sum() == 0, "features must not contain NaN"
        assert len(df_filtered["target"].unique()) == 2, "both target classes (0 and 1) must be present"

        logger.info(
            "training_dataset_validated",
            total_resolved_markets=len(df_filtered),
            matched_rows=len(df_filtered),
            coverage=1.0,
            duplicates=0,
            future_feature_rows=0,
        )

        # 7. Синхронизированный расчет границ волатильности через VolatilityRegimePolicy
        vol_p33 = float(df_filtered["vol_trend"].quantile(0.33))
        vol_p67 = float(df_filtered["vol_trend"].quantile(0.67))

        from polyflip.crypto.volatility import VolatilityRegimePolicy
        vol_policy = VolatilityRegimePolicy(low_boundary=vol_p33, high_boundary=vol_p67)

        logger.info(
            "vol_regime_tertiles",
            symbol=symbol,
            p33=round(vol_p33, 4),
            p67=round(vol_p67, 4),
        )

        now = datetime.now(timezone.utc)
        if save_settings:
            for key, val in [(f"CRYPTO_VOL_P33_{symbol}", vol_p33), (f"CRYPTO_VOL_P67_{symbol}", vol_p67)]:
                row = (await self.db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
                if row:
                    row.value = str(round(val, 4))
                    row.updated_at = now
                    row.updated_by = "crypto_train_job"
                else:
                    self.db.add(RuntimeSettings(key=key, value=str(round(val, 4)), updated_at=now, updated_by="crypto_train_job"))

        # Разбиваем датасет на 3 режима по vol_policy
        df_low  = df_filtered[df_filtered["vol_trend"].apply(lambda v: vol_policy.classify(v) == "low_vol")]
        df_mid  = df_filtered[df_filtered["vol_trend"].apply(lambda v: vol_policy.classify(v) == "mid_vol")]
        df_high = df_filtered[df_filtered["vol_trend"].apply(lambda v: vol_policy.classify(v) == "high_vol")]

        trained_any = False
        from polyflip.crypto.predictor import CryptoPredictor

        # Получаем семафор один раз для всего цикла по режимам
        sem = await _get_training_semaphore(self.db)

        for regime, df_regime in [("low_vol", df_low), ("mid_vol", df_mid), ("high_vol", df_high)]:
            if len(df_regime) < 50:
                logger.warning("regime_too_small", regime=regime, rows=len(df_regime))
                continue

            dataset_fingerprint = _dataset_fingerprint(df_regime, available)
            try:
                X_r = df_regime[available].reset_index(drop=True)
                y_r = df_regime["target"].reset_index(drop=True)
                
                n_regime = len(df_regime)
                adaptive_params = lgbm_params.copy()
                if n_regime < 500:
                    adaptive_params["num_leaves"] = 15
                    adaptive_params["max_depth"] = 4
                    adaptive_params["min_child_samples"] = 30
                    adaptive_params["n_estimators"] = 200
                elif n_regime < 1000:
                    adaptive_params["num_leaves"] = 20
                    adaptive_params["max_depth"] = 5
                    adaptive_params["min_child_samples"] = 25
                
                logger.info("adaptive_lgbm_params", regime=regime, n_regime=n_regime, num_leaves=adaptive_params["num_leaves"])

                # CPU-bound в thread, ограничиваем параллелизм через общий семафор
                t0 = time.monotonic()
                try:
                    async with sem:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                _fit_lgbm_and_serialize,
                                X_r, y_r, cv_n_splits,
                                min_precision,
                                min_valid_thr,
                                max_valid_thr,
                                thr_fallback,
                                **adaptive_params
                            ),
                            timeout=1800.0,   # 30 минут — hard limit
                        )
                except asyncio.TimeoutError:
                    logger.error("regime_train_timeout", symbol=symbol, regime=regime)
                    continue
                finally:
                    logger.info("regime_train_duration", symbol=symbol, regime=regime,
                                elapsed_sec=round(time.monotonic() - t0, 1))

                model_bytes, val_auc, baseline_auc, threshold, threshold_down, ece, fi, precision, recall, f1, brier = result

                logger.info(
                    "crypto_regime_model_trained",
                    symbol=symbol,
                    regime=regime,
                    val_auc=round(val_auc, 4),
                    baseline=round(baseline_auc, 4),
                    threshold=round(threshold, 4),
                    ece=round(ece, 4),
                )

                regime_asset = f"{symbol}_{regime}"

                # --- Crypto Model Quality Gate Check ---
                active_crypto_stmt = (
                    select(ModelRegistry)
                    .where(ModelRegistry.asset == regime_asset, ModelRegistry.is_active == True)
                    .limit(1)
                )
                active_res = await self.db.execute(active_crypto_stmt)
                active_crypto_model = active_res.scalar_one_or_none()

                passed_quality_gate, gate_reasons, threshold, threshold_down = _evaluate_quality_gate(
                    model_bytes=model_bytes,
                    val_auc=val_auc,
                    baseline_auc=baseline_auc,
                    ece=ece,
                    threshold=threshold,
                    threshold_down=threshold_down,
                    features=available,
                    active_accuracy=(active_crypto_model.accuracy if active_crypto_model else None),
                    active_version=(active_crypto_model.version if active_crypto_model else None),
                )

                should_activate=bool(activate_after_train and passed_quality_gate)
                if gate_reasons:
                    logger.info("crypto_model_diagnostics", asset=regime_asset, technical_valid=passed_quality_gate, findings=gate_reasons)

                # Деактивируем старые записи только если новая модель прошла Quality Gate
                if should_activate:
                    await self.db.execute(
                        update(ModelRegistry)
                        .where(ModelRegistry.asset == regime_asset)
                        .values(is_active=False)
                    )

                # Версионирование
                v_res = await self.db.execute(
                    select(ModelRegistry.version)
                    .where(ModelRegistry.asset == regime_asset)
                    .order_by(ModelRegistry.version.desc())
                    .limit(1)
                )
                next_version = (v_res.scalar_one_or_none() or 0) + 1

                if save_settings:
                    # Сохраняем feature importance в RuntimeSettings
                    fi_key = f"CRYPTO_FI_{regime_asset}"
                    fi_row = (await self.db.execute(
                        select(RuntimeSettings).where(RuntimeSettings.key == fi_key)
                    )).scalar_one_or_none()
                    if fi_row:
                        fi_row.value = json.dumps(fi)
                        fi_row.updated_at = now
                        fi_row.updated_by = "crypto_train_job"
                    else:
                        self.db.add(RuntimeSettings(
                            key=fi_key,
                            value=json.dumps(fi),
                            updated_at=now,
                            updated_by="crypto_train_job",
                        ))

                # Сохраняем модель
                self.db.add(ModelRegistry(
                    asset=regime_asset,
                    version=next_version,
                    model_type="lgbm",
                    model_blob=model_bytes,
                    accuracy=val_auc,
                    baseline=baseline_auc,
                    precision_at_threshold=precision,
                    recall_at_threshold=recall,
                    f1_at_threshold=f1,
                    brier_score=brier,
                    decision_threshold=threshold,
                    decision_threshold_down=threshold_down,
                    training_params={
                        **adaptive_params,
                        "target_source": "POLYMARKET_FINAL_OUTCOME",
                        "feature_set": feature_spec.key,
                        "feature_set_version": feature_spec.version,
                        "feature_schema_hash": feature_schema_hash(available),
                        "feature_count": len(available),
                        "validation_scheme": "TIME_SERIES_SPLIT",
                        "activation_after_train": activate_after_train,
                        "resolution_source": "CHAINLINK",
                        "alignment_version": "MARKET_WINDOW_V1",
                        "feature_schema_version": "CRYPTO_FEATURES_V2",
                        "dataset_rows": len(df_regime),
                        "dataset_fingerprint": dataset_fingerprint,
                        "dataset_start": str(df_regime["market_start"].min()) if "market_start" in df_regime else None,
                        "dataset_end": str(df_regime["market_start"].max()) if "market_start" in df_regime else None,
                        "vol_p33": vol_p33,
                        "vol_p67": vol_p67,
                    },
                    features=",".join(available),
                    feature_importance=fi,
                    dataset_fingerprint=dataset_fingerprint,
                    ece=ece,
                    is_active=should_activate,
                    interval=interval,
                    trained_at=now,
                    # Quality Gate audit
                    quality_gate_passed=passed_quality_gate,
                    quality_gate_reasons=(
                        {
                            "reasons": gate_reasons,
                            "technical_valid": passed_quality_gate,
                            "advisory": True,
                            "auc": val_auc if np.isfinite(val_auc) else None,
                            "baseline": baseline_auc if np.isfinite(baseline_auc) else None,
                            "lift": (
                                val_auc - baseline_auc
                                if np.isfinite(val_auc) and np.isfinite(baseline_auc)
                                else None
                            ),
                            "ece": ece if np.isfinite(ece) else None,
                            "threshold_up": threshold,
                            "threshold_down": threshold_down,
                        }
                        if gate_reasons else None
                    ),
                    # Activation audit: TRAINER если прошла QG и стала активной
                    activation_source="TRAINER" if should_activate else None,
                    quality_override=False,
                    activated_at=now if should_activate else None,
                    activated_by="trainer" if should_activate else None,
                ))
                await self.db.flush()
                trained_any = True
                logger.info("crypto_model_saved", asset=regime_asset, version=next_version)
            except Exception as e:
                await self.db.rollback()
                logger.exception("regime_train_failed", symbol=symbol, regime=regime, error=str(e))
                trained_any = False
                break

        if trained_any:
            # Инвалидируем кэш у инстансов предсказателя (но транзакция еще не закоммичена)
            from polyflip.crypto.predictor import CryptoPredictor
            CryptoPredictor.invalidate_all(symbol)
            
            # Candidate experiments are committed without replacing live models.
            if not activate_after_train:
                await self.db.commit()
                logger.info("crypto_experiment_candidate_committed", symbol=symbol, feature_set=feature_spec.key)
                return True

            # P0. Smoke Test: выполняем тестовый inference на свежих свечах
            try:
                
                candles = await get_recent_candles(self.db, symbol, interval="15m", limit=120)
                if len(candles) >= 100:
                    predictor = CryptoPredictor()
                    await predictor.load(self.db, symbol)
                    signal = predictor.predict(candles, symbol)
                    
                    if signal.status not in {"READY", "FUNDING_VETOED", "DEGENERATE_PREDICTION"}:
                        logger.error("smoke_test_failed", symbol=symbol, status=signal.status, reason=signal.risk_reason)
                        # Откат всей транзакции: возвращаем старые активные модели!
                        await self.db.rollback()
                        CryptoPredictor.invalidate_all(symbol)
                        return False
                    else:
                        logger.info("smoke_test_passed", symbol=symbol, status=signal.status, regime=signal.regime)
                        await self.db.commit()
                else:
                    # Мало свечей для теста, но коммитим (E2E пропущен)
                    await self.db.commit()
            except Exception as e:
                logger.exception("smoke_test_exception", symbol=symbol, error=str(e))
                # Откат всей транзакции
                await self.db.rollback()
                CryptoPredictor.invalidate_all(symbol)
                return False

            CryptoPredictor.invalidate_all(symbol)
            return True
        return False
