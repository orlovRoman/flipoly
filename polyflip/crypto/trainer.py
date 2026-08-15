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
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier, early_stopping
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator, calibration_curve
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss, precision_recall_curve, log_loss
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
from polyflip.crypto.feature_audit import (
    feature_audit_summary,
    model_gain_importance,
    summarize_fold_importance,
)
from polyflip.crypto.market_outcome_dataset import build_market_outcome_dataset
from polyflip.db.models import CryptoCandle, ModelRegistry, ModelRegistryOOFArtifact, RuntimeSettings
from polyflip.crypto.polymarket_backtest import compute_oof_polymarket_backtest, load_market_entry_quotes
from polyflip.crypto.oof_artifact import serialize_oof_artifact, OOF_ARTIFACT_SCHEMA_VERSION
from polyflip.crypto.threshold_optimizer import TARGET_COVERAGES, optimize_joint_thresholds

# Импортируем общий семафор из LogReg-трейнера.
# NOTE: Семафор инициализируется один раз при первом вызове и кэшируется до перезапуска сервиса.
# Изменение TRAIN_MAX_PARALLEL_JOBS в RuntimeSettings вступит в силу только после рестарта.
from polyflip.models.trainer import _get_training_semaphore
from polyflip.crypto.feature_sets import CONTROL_FEATURES, feature_schema_hash, get_feature_set
from polyflip.crypto.experiment_configs import normalize_experiment_config, experiment_config_hash

logger = structlog.get_logger(__name__)

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

    if normalized_thresholds[1] >= normalized_thresholds[0]:
        reasons.append(
            "ThresholdCompatibilityError: lower threshold must be strictly below upper threshold"
        )

    smoke_error = _model_smoke_test(model_bytes, features)
    if smoke_error:
        reasons.append(smoke_error)

    technical_errors = [
        reason for reason in reasons
        if reason.startswith(("ModelCompatibilityError", "ThresholdCompatibilityError"))
    ]
    return not technical_errors, reasons, normalized_thresholds[0], normalized_thresholds[1]


# Epsilon filter removed as per MARKET_WINDOW_V1 spec.


@dataclass(frozen=True)
class LGBMFitResult:
    """Named, stable result of one LightGBM fit.

    ``__iter__``/``__getitem__`` preserve the historic eleven-value tuple
    contract for private callers while the trainer uses explicit attributes.
    Optional OOT fields are always present on the object, never appended to a
    tuple conditionally.
    """

    model_bytes: bytes
    val_auc: float
    baseline_auc: float
    threshold: float
    threshold_down: float
    ece: float
    feature_importance: dict[str, int]
    precision: float
    recall: float
    f1: float
    brier: float
    oot_samples: int | None = None
    log_loss: float | None = None
    feature_audit: dict[str, dict[str, float | int]] = field(default_factory=dict)
    feature_audit_summary: dict[str, object] = field(default_factory=dict)
    # OOF probabilities are retained for the canonical Polymarket PnL audit.
    oof_scores: np.ndarray | None = None
    raw_oof_scores: np.ndarray | None = None
    calibration_method: str = "PLATT"
    calibration_comparison: dict[str, dict[str, float | None]] = field(default_factory=dict)
    threshold_sweep: list[dict[str, object]] = field(default_factory=list)
    # Effective LightGBM parameters, including the selected controlled-search trial.
    effective_params: dict[str, object] = field(default_factory=dict)

    def _legacy_values(self) -> tuple[object, ...]:
        return (
            self.model_bytes, self.val_auc, self.baseline_auc,
            self.threshold, self.threshold_down, self.ece,
            self.feature_importance, self.precision, self.recall,
            self.f1, self.brier,
        )

    def __iter__(self):
        return iter(self._legacy_values())

    def __getitem__(self, index):
        return self._legacy_values()[index]

    def __len__(self) -> int:
        return 11


class CalibratedLightGBMModel:
    """Pickle-safe model bundle with separate raw and calibrated scores."""

    def __init__(self, raw_model, calibrated_model, calibration_method: str) -> None:
        self.raw_model = raw_model
        self.calibrated_model = calibrated_model
        self.calibration_method = str(calibration_method).upper()
        self.n_features_in_ = getattr(raw_model, "n_features_in_", None)
        self.feature_importances_ = getattr(raw_model, "feature_importances_", None)

    def predict_raw_proba(self, rows):
        return self.raw_model.predict_proba(rows)

    def predict_proba(self, rows):
        return self.calibrated_model.predict_proba(rows)


def _make_lgbm(**params) -> LGBMClassifier:
    """Вспомогательная функция для создания квалифицированного LGBMClassifier."""
    defaults = {
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        15,
        "max_depth":         4,
        "min_child_samples": 50,
        "subsample":         0.8,
        # LightGBM disables row bagging when subsample_freq=0 (its default).
        # Keep the UI's subsample setting semantically real.
        "subsample_freq":    1,
        "colsample_bytree":  1.0,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      CV_RANDOM_STATE,
        "n_jobs":            2,
        "verbose":          -1,
    }
    defaults.update(params)
    return LGBMClassifier(**defaults)


def _fit_with_early_stopping(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    early_stopping_rounds: int,
) -> LGBMClassifier:
    """Fit a model with a chronological validation tail when it is safe."""
    rounds = max(0, int(early_stopping_rounds))
    if rounds == 0 or len(X) < 100 or y.nunique(dropna=False) < 2:
        model.fit(X, y)
        return model

    validation_size = max(20, int(len(X) * 0.15))
    if validation_size >= len(X) - 20:
        model.fit(X, y)
        return model
    fit_end = len(X) - validation_size
    X_fit, X_eval = X.iloc[:fit_end], X.iloc[fit_end:]
    y_fit, y_eval = y.iloc[:fit_end], y.iloc[fit_end:]
    if y_fit.nunique(dropna=False) < 2 or y_eval.nunique(dropna=False) < 2:
        model.fit(X, y)
        return model
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_eval, y_eval)],
        callbacks=[early_stopping(rounds, verbose=False)],
    )
    return model


def _fit_lgbm_and_serialize(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_N_SPLITS,
    min_precision: float = 0.52,
    min_valid_thr: float = 0.30,
    max_valid_thr: float = 0.75,
    thr_fallback: float = 0.55,
    return_metrics: bool = False,
    early_stopping_rounds: int = 30,
    compute_feature_audit: bool = False,
    calibration_method: str = "PLATT",
    selected_target_coverage: float = 0.40,
    backtest_frame: pd.DataFrame | None = None,
    backtest_quotes: pd.DataFrame | None = None,
    backtest_options: dict | None = None,
    **lgbm_params,
) -> LGBMFitResult:
    """
    CPU-bound. Обучает LightGBM с TimeSeriesSplit.
    Returns a named LGBMFitResult with stable OOT audit fields.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_scores = np.full(len(y), np.nan)
    raw_oof_scores = np.full(len(y), np.nan)
    requested_calibration = str(calibration_method or "PLATT").strip().upper()
    if requested_calibration in {"SIGMOID", "PLATT"}:
        calibration_methods = ("PLATT",)
    elif requested_calibration == "ISOTONIC":
        calibration_methods = ("ISOTONIC",)
    elif requested_calibration == "NONE":
        calibration_methods = ("NONE",)
    else:
        # AUTO is intentionally conservative: isotonic must beat PLATT on all
        # available OOT diagnostics before it is allowed to win.
        calibration_methods = ("PLATT", "ISOTONIC")
    calibration_oof = {
        method: np.full(len(y), np.nan) for method in calibration_methods
    }
    aucs: list[float] = []
    fold_gain_importances: list[np.ndarray] = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_lgbm = _make_lgbm(**lgbm_params)
        _fit_with_early_stopping(fold_lgbm, X_train, y_train, early_stopping_rounds)
        if compute_feature_audit:
            fold_gain_importances.append(model_gain_importance(fold_lgbm, X.columns))

        raw_prob = np.asarray(fold_lgbm.predict_proba(X.iloc[val_idx])[:, 1], dtype=float)
        raw_oof_scores[val_idx] = raw_prob
        # Calibrate on the first half of the validation tail and score only the
        # second half. The raw score remains the source of direction.
        mid = len(val_idx) // 2
        cal_idx, eval_idx = val_idx[:mid], val_idx[mid:]

        if len(cal_idx) >= 20 and len(eval_idx) >= 20:
            X_cal,  y_cal  = X.iloc[cal_idx],  y.iloc[cal_idx]
            X_eval, y_eval = X.iloc[eval_idx], y.iloc[eval_idx]
            for method in calibration_methods:
                if method == "NONE":
                    y_proba = raw_prob[eval_idx - val_idx[0]]
                else:
                    fold_cal = CalibratedClassifierCV(
                        estimator=FrozenEstimator(fold_lgbm),
                        method="sigmoid" if method == "PLATT" else "isotonic",
                        cv=None,
                    )
                    fold_cal.fit(X_cal, y_cal)
                    y_proba = np.asarray(fold_cal.predict_proba(X_eval)[:, 1], dtype=float)
                calibration_oof[method][eval_idx] = y_proba
            y_proba = calibration_oof[calibration_methods[0]][eval_idx]
            oof_scores[eval_idx] = y_proba
            aucs.append(roc_auc_score(y_eval, y_proba))
        else:
            # Small folds cannot support a reliable calibrator.
            for method in calibration_methods:
                calibration_oof[method][val_idx] = raw_prob
            oof_scores[val_idx] = raw_prob
            aucs.append(roc_auc_score(y.iloc[val_idx], raw_prob))

    val_auc = float(np.mean(aucs))
    baseline_auc = 0.5

    def _ece(y_true: np.ndarray, probs: np.ndarray) -> float:
        if len(y_true) <= 10:
            return 0.5
        try:
            frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
            edges = np.linspace(0.0, 1.0, 11)
            bucket = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, 9)
            counts = np.bincount(bucket, minlength=10)
            weights = counts[counts > 0] / len(probs)
            return float(np.sum(weights * np.abs(frac_pos - mean_pred)))
        except ValueError:
            return 0.5

    calibration_comparison: dict[str, dict[str, float | None]] = {}
    for method, scores in calibration_oof.items():
        valid = np.isfinite(scores)
        if valid.sum() == 0:
            continue
        y_valid = y[valid].to_numpy(dtype=float)
        p_valid = scores[valid]
        try:
            method_log_loss = float(log_loss(y_valid, p_valid, labels=[0, 1]))
        except ValueError:
            method_log_loss = None
        method_pnl = None
        if backtest_frame is not None and backtest_quotes is not None and not backtest_quotes.empty:
            try:
                economic = compute_oof_polymarket_backtest(
                    backtest_frame,
                    scores,
                    backtest_quotes,
                    strategy_branch="COMBINED",
                    **(backtest_options or {}),
                )
                method_pnl = float(economic.get("net_profit") or 0.0)
            except Exception:
                method_pnl = None
        calibration_comparison[method] = {
            "brier": float(brier_score_loss(y_valid, p_valid)),
            "ece": _ece(y_valid, p_valid),
            "log_loss": method_log_loss,
            "polymarket_pnl": method_pnl,
        }

    selected_method = requested_calibration
    if selected_method in {"AUTO", "SIGMOID"}:
        selected_method = "PLATT"
    if requested_calibration == "AUTO" and "ISOTONIC" in calibration_comparison:
        platt = calibration_comparison.get("PLATT", {})
        isotonic = calibration_comparison["ISOTONIC"]
        pnl_ok = (
            platt.get("polymarket_pnl") is None
            or isotonic.get("polymarket_pnl") is None
            or isotonic["polymarket_pnl"] >= platt.get("polymarket_pnl", -np.inf)
        )
        if (
            isotonic.get("brier", np.inf) <= platt.get("brier", np.inf)
            and isotonic.get("ece", np.inf) <= platt.get("ece", np.inf)
            and pnl_ok
            and (
                isotonic.get("brier", np.inf) < platt.get("brier", np.inf)
                or isotonic.get("ece", np.inf) < platt.get("ece", np.inf)
                or isotonic.get("polymarket_pnl", -np.inf) > platt.get("polymarket_pnl", -np.inf)
            )
        ):
            selected_method = "ISOTONIC"
    if selected_method not in calibration_oof:
        selected_method = calibration_methods[0]
    oof_scores = calibration_oof[selected_method].copy()
    valid_mask = np.isfinite(oof_scores)
    ece = calibration_comparison.get(selected_method, {}).get("ece", 0.5) or 0.5
    logger.info(
        "crypto_calibration",
        method=selected_method,
        ece=round(float(ece), 4),
        comparison=calibration_comparison,
    )

    # Финальная модель на всех данных
    n_cal = max(20, int(len(X) * 0.15))
    n_cal = min(n_cal, max(20, len(X) // 3))
    X_fit, X_cal_final = X.iloc[:-n_cal], X.iloc[-n_cal:]
    y_fit, y_cal_final = y.iloc[:-n_cal], y.iloc[-n_cal:]

    # Probe the chronological tail only to determine the stopping point,
    # then refit the selected number of trees on all of X_fit.  The previous
    # implementation kept the probe's 85% sub-fit as the production artifact.
    probe_lgbm = _make_lgbm(**lgbm_params)
    _fit_with_early_stopping(probe_lgbm, X_fit, y_fit, early_stopping_rounds)
    best_iteration = getattr(probe_lgbm, "best_iteration_", None)
    final_params = dict(lgbm_params)
    if isinstance(best_iteration, (int, np.integer)) and int(best_iteration) > 0:
        final_params["n_estimators"] = int(best_iteration)
    final_lgbm = _make_lgbm(**final_params)
    final_lgbm.fit(X_fit, y_fit)
    final_cal = final_lgbm
    if selected_method != "NONE":
        try:
            final_cal = CalibratedClassifierCV(
                estimator=FrozenEstimator(final_lgbm),
                method="sigmoid" if selected_method == "PLATT" else "isotonic",
                cv=None,
            )
            final_cal.fit(X_cal_final, y_cal_final)
        except ValueError:
            logger.warning("final_calibration_failed_fallback_raw", method=selected_method)
            selected_method = "NONE"
            final_cal = final_lgbm
    model_bundle = CalibratedLightGBMModel(final_lgbm, final_cal, selected_method)

    valid_mask = np.isfinite(oof_scores) & np.isfinite(raw_oof_scores)
    threshold_sweep: list[dict[str, object]] = []
    if backtest_frame is not None and backtest_quotes is not None and not backtest_quotes.empty:
        threshold_audit = optimize_joint_thresholds(
            backtest_frame,
            raw_oof_scores,
            oof_scores,
            backtest_quotes,
            target_coverages=TARGET_COVERAGES,
            selected_target_coverage=selected_target_coverage,
            **(backtest_options or {}),
        )
        optimal_threshold = float(threshold_audit["selected_lower_threshold"])
        optimal_threshold_up = float(threshold_audit["selected_upper_threshold"])
        threshold_sweep = list(threshold_audit.get("sweep") or [])
    else:
        valid_raw = raw_oof_scores[valid_mask]
        target = max(0.05, min(0.95, float(selected_target_coverage)))
        optimal_threshold = float(np.quantile(valid_raw, target / 2.0)) if len(valid_raw) else 0.45
        optimal_threshold_up = float(np.quantile(valid_raw, 1.0 - target / 2.0)) if len(valid_raw) else 0.55
        if optimal_threshold >= optimal_threshold_up:
            optimal_threshold, optimal_threshold_up = 0.49, 0.51

    # Direction precision is retained as a diagnostic only; it no longer
    # chooses either threshold.
    valid_oof = oof_scores[valid_mask]
    direction_oof = raw_oof_scores[valid_mask]
    y_valid = y[valid_mask]
    predictions_at_thr = (direction_oof <= optimal_threshold) | (direction_oof >= optimal_threshold_up)
    if predictions_at_thr.sum() > 20:
        real_precision = float((y_valid[predictions_at_thr] == 1).mean())
        signal_rate = float(predictions_at_thr.mean())
        logger.info("oof_real_precision",
            precision=round(real_precision, 4),
            signal_rate=round(signal_rate, 4),
            threshold=round(optimal_threshold_up, 4))
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
    if compute_feature_audit:
        feature_audit = summarize_fold_importance(fold_gain_importances, tuple(X.columns))
        audit_summary = feature_audit_summary(feature_audit)
        logger.info(
            "crypto_feature_importance_stability",
            stable_features=audit_summary["stable_features"],
            zero_gain_features=audit_summary["zero_gain_features"],
        )
    else:
        feature_audit = {}
        audit_summary = {}

    y_oof = y[valid_mask].astype(int)
    p_oof = oof_scores[valid_mask]
    # Registry precision/recall remain a directional UP diagnostic.  The
    # actionable trade signal itself is evaluated jointly above; using the
    # lower threshold here would incorrectly label the entire dead zone as UP.
    y_pred = (direction_oof >= optimal_threshold_up).astype(int)

    precision = float(precision_score(y_oof, y_pred, zero_division=0))
    recall = float(recall_score(y_oof, y_pred, zero_division=0))
    f1_metric = float(f1_score(y_oof, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_oof, p_oof))
    try:
        oot_log_loss = float(log_loss(y_oof, p_oof, labels=[0, 1]))
    except ValueError:
        oot_log_loss = None

    return LGBMFitResult(
        model_bytes=pickle.dumps(model_bundle),
        val_auc=val_auc,
        baseline_auc=baseline_auc,
        threshold=optimal_threshold_up,
        threshold_down=optimal_threshold,
        ece=ece,
        feature_importance=fi,
        precision=precision,
        recall=recall,
        f1=f1_metric,
        brier=brier,
        oot_samples=int(valid_mask.sum()) if return_metrics else None,
        log_loss=oot_log_loss if return_metrics else None,
        feature_audit=feature_audit,
        feature_audit_summary=audit_summary,
        oof_scores=oof_scores.copy() if return_metrics else None,
        raw_oof_scores=raw_oof_scores.copy() if return_metrics else None,
        calibration_method=selected_method,
        calibration_comparison=calibration_comparison,
        threshold_sweep=threshold_sweep,
        effective_params={
            **final_params,
            "n_estimators": int(getattr(final_lgbm, "n_estimators_", final_params.get("n_estimators", 0))),
        },
    )


def _controlled_lgbm_candidates(base_params: dict, trials: int) -> list[dict]:
    """Return a small deterministic search space, never an unbounded sweep."""
    base = dict(base_params)
    candidates = [base]
    variants = (
        {"learning_rate": float(base.get("learning_rate", 0.05)) * 0.7,
         "num_leaves": max(7, int(base.get("num_leaves", 15) * 0.75))},
        {"learning_rate": float(base.get("learning_rate", 0.05)) * 1.3,
         "num_leaves": min(128, int(base.get("num_leaves", 15) * 1.25)),
         "min_child_samples": max(10, int(base.get("min_child_samples", 50) * 0.8))},
        {"reg_lambda": float(base.get("reg_lambda", 1.0)) * 2.0,
         "min_child_samples": max(10, int(base.get("min_child_samples", 50) * 1.25))},
    )
    for changes in variants[: max(0, min(int(trials), 4) - 1)]:
        candidate = dict(base)
        candidate.update(changes)
        candidates.append(candidate)
    return candidates


def _fit_controlled_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    search_trials: int,
    early_stopping_rounds: int,
    n_splits: int,
    min_precision: float,
    min_valid_thr: float,
    max_valid_thr: float,
    thr_fallback: float,
    base_params: dict,
    backtest_frame: pd.DataFrame | None = None,
    backtest_quotes: pd.DataFrame | None = None,
    backtest_options: dict | None = None,
    calibration_method: str = "PLATT",
    selected_target_coverage: float = 0.40,
) -> LGBMFitResult:
    candidates = _controlled_lgbm_candidates(base_params, search_trials)
    if len(candidates) == 1:
        return _fit_lgbm_and_serialize(
            X,
            y,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            min_precision=min_precision,
            min_valid_thr=min_valid_thr,
            max_valid_thr=max_valid_thr,
            thr_fallback=thr_fallback,
            return_metrics=True,
            compute_feature_audit=True,
            calibration_method=calibration_method,
            selected_target_coverage=selected_target_coverage,
            backtest_frame=backtest_frame,
            backtest_quotes=backtest_quotes,
            backtest_options=backtest_options,
            **candidates[0],
        )
    trial_results = []
    for params in candidates:
        trial_results.append(_fit_lgbm_and_serialize(
            X,
            y,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            min_precision=min_precision,
            min_valid_thr=min_valid_thr,
            max_valid_thr=max_valid_thr,
            thr_fallback=thr_fallback,
            return_metrics=True,
            compute_feature_audit=False,
            calibration_method=calibration_method,
            selected_target_coverage=selected_target_coverage,
            backtest_frame=backtest_frame,
            backtest_quotes=backtest_quotes,
            backtest_options=backtest_options,
            **params,
        ))

    def score(result: LGBMFitResult) -> float:
        # Prefer robust Polymarket economics when quotes are available.  A
        # candidate with fewer than 50 OOT trades is retained for diagnostics
        # but cannot outrank a candidate that meets the coverage floor.
        if (
            backtest_frame is not None
            and backtest_quotes is not None
            and not backtest_quotes.empty
        ):
            options = dict(backtest_options or {})
            branch_result = compute_oof_polymarket_backtest(
                backtest_frame,
                result.oof_scores,
                backtest_quotes,
                strategy_branch="COMBINED",
                **options,
            )
            trades = int(branch_result.get("n_trades") or 0)
            windows = branch_result.get("oot_windows") or []
            median_window_pnl = float(np.median([
                float(window.get("net_profit") or 0.0) for window in windows
            ])) if windows else float(branch_result.get("net_profit") or 0.0)
            max_dd = float(branch_result.get("max_drawdown_usdc") or 0.0)
            window_pnls = [
                float(window.get("net_profit") or 0.0)
                for window in windows
                if isinstance(window, dict)
            ]
            # A median alone can hide a failed time slice. Penalize the
            # negative tail and require all three chronological OOT windows
            # before a trial can be considered stable.
            worst_window_loss = max(0.0, -min(window_pnls)) if window_pnls else 0.0
            economic_score = median_window_pnl - 0.5 * max_dd - 0.25 * worst_window_loss
            if len(window_pnls) < 3:
                economic_score -= 10.0 * (3 - len(window_pnls))
            if trades < 50:
                economic_score -= 0.05 * (50 - trades)
            return economic_score
        brier = result.brier if np.isfinite(result.brier) else 1.0
        return float(result.val_auc - brier)

    best_index = max(range(len(trial_results)), key=lambda index: score(trial_results[index]))
    best_params = candidates[best_index]
    logger.info(
        "lgbm_hyperparameter_search_selected",
        trials=len(candidates),
        selected_trial=best_index + 1,
        scores=[round(score(result), 6) for result in trial_results],
        objective=(
            "median_oot_polymarket_pnl_minus_drawdown"
            if backtest_frame is not None and backtest_quotes is not None and not backtest_quotes.empty
            else "auc_minus_brier"
        ),
        min_oot_trades=50,
    )
    return _fit_lgbm_and_serialize(
        X,
        y,
        n_splits=n_splits,
        early_stopping_rounds=early_stopping_rounds,
        min_precision=min_precision,
        min_valid_thr=min_valid_thr,
        max_valid_thr=max_valid_thr,
        thr_fallback=thr_fallback,
        return_metrics=True,
        compute_feature_audit=True,
        calibration_method=calibration_method,
        selected_target_coverage=selected_target_coverage,
        backtest_frame=backtest_frame,
        backtest_quotes=backtest_quotes,
        backtest_options=backtest_options,
        **best_params,
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


async def _get_string_setting(db: AsyncSession, key: str, default: str = "") -> str:
    row = (await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
    return str(row.value).strip() if row and row.value is not None else default


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
        activate_after_train: bool = False,
        experiment_config: dict | None = None,
        experiment_config_id: int | None = None,
    ) -> bool:
        normalized_config = normalize_experiment_config(experiment_config) if experiment_config else None
        if normalized_config:
            feature_set = normalized_config["feature_set"]
        feature_spec = get_feature_set(feature_set)
        available = list(feature_spec.features)
        n_jobs = await _get_int_setting(self.db, "CRYPTO_LGBM_N_JOBS", 2)
        early_stopping_rounds = await _get_int_setting(self.db, "CRYPTO_LGBM_EARLY_STOPPING_ROUNDS", 30)
        search_trials = await _get_int_setting(self.db, "CRYPTO_LGBM_HYPERPARAM_SEARCH_TRIALS", 1)
        calibration_method = await _get_string_setting(self.db, "LGBM_CALIBRATION_METHOD", "PLATT")
        selected_target_coverage = await _get_float_setting(self.db, "LGBM_TARGET_COVERAGE", 0.40)

        logger.info(
            "crypto_training_start",
            symbol=symbol,
            interval=interval,
            feature_set=feature_spec.key,
            feature_set_version=feature_spec.version,
            activate_after_train=activate_after_train,
        )

        # Загружаем все свечи для символа
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
            "subsample_freq": 1 if subsample < 1.0 else 0,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "n_jobs": max(1, min(n_jobs, 32)),
        }
        if normalized_config:
            lgbm_params.update(normalized_config["model"])
            calibration_method = normalized_config["calibration"]["method"]
            selected_target_coverage = float(normalized_config["thresholds"]["target_coverage"])
            logger.info(
                "lgbm_experiment_config_applied",
                experiment_config_id=experiment_config_id,
                config_hash=experiment_config_hash(normalized_config),
            )

        min_precision = await get_float(self.db, "LGBM_MIN_PRECISION_FOR_THRESHOLD")
        min_valid_thr = await get_float(self.db, "LGBM_MIN_VALID_THRESHOLD")
        max_valid_thr = await get_float(self.db, "LGBM_MAX_VALID_THRESHOLD")
        thr_fallback = await get_float(self.db, "LGBM_THRESHOLD_FALLBACK")
        cv_n_splits = await get_int(self.db, "LGBM_CV_N_SPLITS")
        epsilon_quantile = await get_float(self.db, "LGBM_EPSILON_QUANTILE")
        # 3. Загружаем выравненный торговый датасет на канонических исходах Polymarket (MARKET_WINDOW_V1)
        df_filtered = await build_market_outcome_dataset(
            self.db, symbol=symbol, interval=interval, feature_set=feature_spec.key
        )
        
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

        # The registry PnL fields are populated from the same canonical
        # Polymarket data, using only OOF probabilities and executable entry
        # quotes. This keeps A/B/C comparable and prevents live-trade absence
        # from being mistaken for a zero-quality backtest.
        backtest_min_edge = await _get_float_setting(self.db, "BACKTEST_MIN_EDGE", 0.04)
        backtest_cost_buffer = await _get_float_setting(self.db, "COMBINED_COST_BUFFER", 0.02)
        backtest_fee_rate = await _get_float_setting(self.db, "POLYMARKET_FEE_RATE", 0.002)
        backtest_min_price = await _get_float_setting(self.db, "TRADE_MIN_PRICE", 0.05)
        backtest_max_price = await _get_float_setting(self.db, "TRADE_MAX_PRICE", 0.95)
        backtest_outsider_max = await _get_float_setting(self.db, "OUTSIDER_MAX_PRICE", 0.45)
        backtest_stake_usdc = 1.0
        backtest_slippage_pct = 0.0
        if normalized_config:
            backtest_params = normalized_config["backtest"]
            backtest_min_edge = float(backtest_params["min_edge"])
            backtest_cost_buffer = float(backtest_params["cost_buffer"])
            backtest_fee_rate = float(backtest_params["fee_rate"])
            backtest_min_price = float(backtest_params["min_price"])
            backtest_max_price = float(backtest_params["max_price"])
            backtest_outsider_max = float(backtest_params["outsider_max_price"])
            backtest_stake_usdc = float(backtest_params["stake_usdc"])
            backtest_slippage_pct = float(backtest_params["slippage_pct"])
        try:
            entry_quotes = await load_market_entry_quotes(
                self.db, df_filtered[["market_id", "market_start"]]
            )
        except Exception:
            logger.exception("lgbm_backtest_quote_load_failed", symbol=symbol)
            entry_quotes = pd.DataFrame()
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
            market_times = pd.to_datetime(df_regime["market_start"], utc=True)
            training_window_start = market_times.min()
            training_window_end = market_times.max()
            # The comparison window is day-granular so the first six rows
            # dropped by sequence features do not split otherwise identical
            # A/B/C experiments into separate groups. Exact timestamps and
            # the feature-specific fingerprint remain available for auditing.
            comparison_key = "|".join((
                f"{symbol}_{regime}",
                "POLYMARKET_FINAL_OUTCOME",
                "TIME_SERIES_SPLIT",
                "LIGHTGBM_TIME_SERIES_SPLIT",
                interval,
                training_window_start.floor("D").isoformat(),
                training_window_end.floor("D").isoformat(),
            ))
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
                                _fit_controlled_lgbm,
                                X_r,
                                y_r,
                                search_trials=search_trials,
                                early_stopping_rounds=early_stopping_rounds,
                                n_splits=cv_n_splits,
                                min_precision=min_precision,
                                min_valid_thr=min_valid_thr,
                                max_valid_thr=max_valid_thr,
                                thr_fallback=thr_fallback,
                                base_params=adaptive_params,
                                backtest_frame=df_regime.reset_index(drop=True),
                                backtest_quotes=entry_quotes,
                                backtest_options={
                                    "min_edge": backtest_min_edge,
                                    "cost_buffer": backtest_cost_buffer,
                                    "fee_rate": backtest_fee_rate,
                                    "min_price": backtest_min_price,
                                    "max_price": backtest_max_price,
                                    "outsider_max_price": backtest_outsider_max,
                                    "stake_usdc": backtest_stake_usdc,
                                    "slippage_pct": backtest_slippage_pct,
                                },
                                calibration_method=calibration_method,
                                selected_target_coverage=selected_target_coverage,
                            ),
                            timeout=1800.0,   # 30 минут — hard limit
                        )
                except asyncio.TimeoutError:
                    logger.error("regime_train_timeout", symbol=symbol, regime=regime)
                    continue
                finally:
                    logger.info("regime_train_duration", symbol=symbol, regime=regime,
                                elapsed_sec=round(time.monotonic() - t0, 1))

                model_bytes = result.model_bytes
                val_auc = result.val_auc
                baseline_auc = result.baseline_auc
                threshold = result.threshold
                threshold_down = result.threshold_down
                ece = result.ece
                fi = result.feature_importance
                precision = result.precision
                recall = result.recall
                f1 = result.f1
                brier = result.brier
                effective_params = result.effective_params or adaptive_params
                fit_metrics = {
                    "oot_samples": result.oot_samples,
                    "brier_score": result.brier,
                    "log_loss": result.log_loss,
                    "feature_audit": result.feature_audit,
                    "feature_audit_summary": result.feature_audit_summary,
                    "calibration_method": result.calibration_method,
                    "calibration_comparison": result.calibration_comparison,
                    "threshold_sweep": result.threshold_sweep,
                }

                # Evaluate all three trading branches only when OOF scores are
                # available. A metrics-disabled fit must not be represented as
                # a real zero-PnL backtest in the registry/dashboard.
                branches = ("OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED")
                if result.oof_scores is None:
                    logger.warning(
                        "lgbm_backtest_skipped_no_oof_scores",
                        symbol=symbol,
                        regime=regime,
                    )
                    backtest_variants = {branch: {} for branch in branches}
                    backtest_outsider = {}
                else:
                    backtest_variants = {}
                    for branch in branches:
                        branch_result = compute_oof_polymarket_backtest(
                            df_regime.reset_index(drop=True),
                            result.oof_scores,
                            entry_quotes,
                            strategy_branch=branch,
                            min_edge=backtest_min_edge,
                            cost_buffer=backtest_cost_buffer,
                            fee_rate=backtest_fee_rate,
                            min_price=backtest_min_price,
                            max_price=backtest_max_price,
                            outsider_max_price=backtest_outsider_max,
                            stake_usdc=backtest_stake_usdc,
                            slippage_pct=backtest_slippage_pct,
                        )
                        backtest_variants[branch] = {
                            key: value for key, value in branch_result.items()
                            if key != "trades"
                        }
                        logger.info(
                            "crypto_lgbm_polymarket_backtest",
                            symbol=symbol,
                            regime=regime,
                            feature_set=feature_spec.key,
                            strategy_branch=branch,
                            trades=branch_result["n_trades"],
                            pnl=round(branch_result["net_profit"], 6),
                            coverage=branch_result["coverage_pct"],
                        )
                    # Registry scalar fields stay backward compatible and point
                    # to the default outsider branch.
                    backtest_outsider = backtest_variants["OUTSIDER_ONLY"]
                fit_metrics["backtest"] = backtest_outsider

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

                should_activate = bool(activate_after_train and passed_quality_gate)
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
                model_row = ModelRegistry(
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
                        **effective_params,
                        "early_stopping_rounds": early_stopping_rounds,
                        "hyperparameter_search_trials": search_trials,
                        "target_source": "POLYMARKET_FINAL_OUTCOME",
                        "feature_set": feature_spec.key,
                        "feature_set_version": feature_spec.version,
                        "feature_schema_hash": feature_schema_hash(available),
                        "feature_count": len(available),
                        "validation_scheme": "TIME_SERIES_SPLIT",
                        "validation_folds": cv_n_splits,
                        "backtest_strategy_branch": "LIGHTGBM_TIME_SERIES_SPLIT",
                        "comparison_key": comparison_key,
                        "activate_after_train": activate_after_train,
                        "experiment_config_id": experiment_config_id,
                        "experiment_config_hash": (
                            experiment_config_hash(normalized_config)
                            if normalized_config else None
                        ),
                        "experiment_config": normalized_config,
                        # Persist the effective values even for legacy/ad-hoc runs so a saved candidate can be replayed exactly.
                        "backtest_config": {
                            "min_edge": backtest_min_edge,
                            "cost_buffer": backtest_cost_buffer,
                            "fee_rate": backtest_fee_rate,
                            "min_price": backtest_min_price,
                            "max_price": backtest_max_price,
                            "outsider_max_price": backtest_outsider_max,
                            "stake_usdc": backtest_stake_usdc,
                            "slippage_pct": backtest_slippage_pct,
                        },
                        "resolution_source": "CHAINLINK",
                        "alignment_version": "MARKET_WINDOW_V1",
                        "feature_schema_version": "CRYPTO_FEATURES_V2",
                        "dataset_rows": len(df_regime),
                        "dataset_markets": len(df_regime),
                        "dataset_fingerprint": dataset_fingerprint,
                        "dataset_start": training_window_start.isoformat(),
                        "dataset_end": training_window_end.isoformat(),
                        "oot_samples": fit_metrics.get("oot_samples"),
                        # One canonical dataset row represents one market;
                        # only the OOF rows are counted as OOT observations.
                        "oot_markets": fit_metrics.get("oot_samples"),
                        "brier_score": fit_metrics.get("brier_score", brier),
                        "log_loss": fit_metrics.get("log_loss"),
                        "feature_audit_version": fit_metrics.get("feature_audit_summary", {}).get("version"),
                        "feature_audit_summary": fit_metrics.get("feature_audit_summary", {}),
                        "feature_selection_policy": "diagnostic_only",
                        "model_config": effective_params,
                        "vol_p33": vol_p33,
                        "vol_p67": vol_p67,
                        "backtest_pnl_mode": "POLYMARKET_OOF",
                        "backtest": backtest_outsider,
                        "backtest_variants": backtest_variants,
                        "calibration_method": result.calibration_method,
                        "calibration_comparison": result.calibration_comparison,
                        "threshold_sweep": result.threshold_sweep,
                        "target_coverage": selected_target_coverage,
                    },
                    features=",".join(available),
                    feature_importance=fi,
                    train_samples=len(df_regime),
                    validation_samples=fit_metrics.get("oot_samples"),
                    positive_rate=float(y_r.mean()),
                    training_window_start=training_window_start.to_pydatetime(),
                    training_window_end=training_window_end.to_pydatetime(),
                    dataset_fingerprint=dataset_fingerprint,
                    ece=ece,
                    backtest_pnl=(
                        float(backtest_outsider["net_profit"])
                        if backtest_outsider else None
                    ),
                    backtest_trades=(
                        int(backtest_outsider["n_trades"])
                        if backtest_outsider else None
                    ),
                    backtest_wr=(
                        float(backtest_outsider["win_rate"])
                        if backtest_outsider else None
                    ),
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
                            "oot_metrics": fit_metrics,
                        }
                        if gate_reasons else None
                    ),
                    # Activation audit: TRAINER если прошла QG и стала активной
                    activation_source="TRAINER" if should_activate else None,
                    quality_override=False,
                    activated_at=now if should_activate else None,
                    activated_by="trainer" if should_activate else None,
                )
                self.db.add(model_row)
                await self.db.flush()
                if result.oof_scores is not None:
                    artifact_blob = serialize_oof_artifact(
                        df_regime.reset_index(drop=True),
                        result.oof_scores,
                        entry_quotes,
                        feature_set=feature_spec.key,
                        feature_schema_hash=feature_schema_hash(available),
                        raw_scores=result.raw_oof_scores,
                    )
                    self.db.add(ModelRegistryOOFArtifact(
                        model_registry_id=model_row.id,
                        schema_version=OOF_ARTIFACT_SCHEMA_VERSION,
                        row_count=len(df_regime),
                        artifact_blob=artifact_blob,
                        created_at=now,
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
