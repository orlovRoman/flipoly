"""
polyflip/models/outsider_trainer.py

Unified training pipeline for Outsider Model A and Model B (Items 2.6 and 2.7).
Enforces:
1. Exact feature contracts: Model A1 (4 features) and Model B1 (7 features).
2. Symmetry: mirror YES/NO outsider situations yield identical Model A probabilities.
3. Strict Stage 1 pipeline standards:
   - SimpleImputer(strategy="median") inside pipeline
   - class_weight=None (unbiased calibration)
   - market_balanced_weights(groups) (equal influence per market)
   - Inner C grid [0.1, 0.5, 1.0] evaluated by log loss
   - Canonical probability metrics (Brier, LogLoss, ECE with 20 bins)
4. Paired delta metrics: delta Brier, delta LogLoss, delta ECE, and delta PnL on identical folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
try:
    from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    from sklearn.calibration import CalibratedClassifierCV
    FrozenEstimator = None
    HAS_FROZEN_ESTIMATOR = False
import structlog

from polyflip.models.probability_metrics import (
    brier_score,
    log_loss_score,
    expected_calibration_error,
)
from polyflip.models.temporal_validation import (
    market_balanced_weights,
    grouped_walk_forward_folds,
)
from polyflip.models.trainer import _group_holdout_indices
from polyflip.models.outsider_feature_sets import (
    MODEL_A1_FEATURES,
    MODEL_B1_FEATURES,
    get_outsider_feature_set,
)
from polyflip.models.outsider_baselines import (
    MarketPriceBaseline,
    evaluate_outsider_predictions,
)

logger = structlog.get_logger(__name__)

C_GRID: tuple[float, ...] = (0.1, 0.5, 1.0)


def build_outsider_logreg_pipeline(c_value: float = 1.0) -> Pipeline:
    """Builds standard scikit-learn pipeline for outsider logistic regression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="lbfgs",
                class_weight=None,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])


@dataclass
class OutsiderTrainResult:
    feature_set: str
    feature_names: tuple[str, ...]
    best_c: float
    oof_predictions: np.ndarray
    metrics: dict[str, Any]
    fold_models: list[Any] = field(default_factory=list)
    final_model: Any = None
    c_selected_per_fold: list[float] = field(default_factory=list)


def train_outsider_model(
    df: pd.DataFrame,
    feature_set: str = "MODEL_A1",
    custom_features: tuple[str, ...] | None = None,
    y_col: str = "target",
    group_col: str = "market_id",
    fold_col: str = "fold",
    fee_rate: float = 0.002,
    min_edge: float = 0.02,
    validation_mode: str = "walk_forward",
    n_splits: int = 5,
) -> OutsiderTrainResult:
    """
    Trains outsider model with grouped cross-validation, inner C search, and honest calibration.
    """
    if custom_features is not None:
        feats = tuple(custom_features)
    else:
        f_spec = get_outsider_feature_set(feature_set)
        feats = f_spec.features

    missing_cols = [f for f in feats if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features in dataframe: {missing_cols}")

    X = df[list(feats)].copy().reset_index(drop=True)
    y = pd.Series(df[y_col].values, dtype=int)
    groups = pd.Series(df[group_col].values)

    n_samples = len(df)
    oof_probs = np.full(n_samples, np.nan, dtype=float)
    fold_models = []
    fold_c_selected = []

    # Outer cross-validation splits
    splits = []
    if validation_mode == "walk_forward":
        ts_col = "decision_at" if "decision_at" in df.columns else ("recorded_at" if "recorded_at" in df.columns else None)
        if ts_col is not None:
            ts_series = df[ts_col]
        else:
            ts_series = pd.date_range("2026-01-01", periods=n_samples, freq="1min", tz="UTC")
        gwf_folds = grouped_walk_forward_folds(groups, ts_series, n_splits=n_splits)
        for gf in gwf_folds:
            if len(gf.train_index) > 0 and len(gf.validation_index) > 0:
                splits.append((gf.train_index, gf.validation_index))
    elif fold_col in df.columns and df[fold_col].nunique() > 1:
        folds = sorted(df[fold_col].unique())
        for f_id in folds:
            val_idx = np.where(df[fold_col].to_numpy() == f_id)[0]
            train_idx = np.where(df[fold_col].to_numpy() != f_id)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

    if not splits:
        if validation_mode == "walk_forward":
            raise ValueError("INSUFFICIENT_HISTORY: Insufficient chronological blocks for walk-forward validation")
        # Fallback to simple 80/20 train/validation split only for non-research mode
        n_train = max(int(n_samples * 0.8), 1)
        splits = [(np.arange(n_train), np.arange(n_train, n_samples))]

    for train_idx, val_idx in splits:
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        groups_train = groups.iloc[train_idx].reset_index(drop=True)

        if len(np.unique(y_train)) < 2:
            continue

        train_weights = market_balanced_weights(groups_train)

        # Split inner train into base fit and inner calibration holdout
        try:
            base_idx, cal_idx = _group_holdout_indices(
                X_train, y_train, groups_train, None, validation_fraction=0.25
            )
            if len(np.unique(y_train.iloc[base_idx])) < 2:
                base_idx = np.arange(len(X_train))
                cal_idx = None
        except Exception:
            base_idx = np.arange(len(X_train))
            cal_idx = None

        # Inner grid search over C_GRID = [0.1, 0.5, 1.0] evaluated on log loss
        fold_best_c = 1.0
        if base_idx is not None and cal_idx is not None and len(cal_idx) > 0:
            c_losses = {}
            for c_cand in C_GRID:
                cand_pipe = build_outsider_logreg_pipeline(c_cand)
                try:
                    cand_pipe.fit(
                        X_train.iloc[base_idx],
                        y_train.iloc[base_idx],
                        model__sample_weight=train_weights[base_idx],
                    )
                    cand_probs = cand_pipe.predict_proba(X_train.iloc[cal_idx])[:, 1]
                    c_losses[c_cand] = log_loss_score(
                        y_train.iloc[cal_idx],
                        cand_probs,
                        sample_weight=train_weights[cal_idx],
                    )
                except Exception:
                    pass
            if c_losses:
                # Item 1.16: Upon tie, pick smaller C
                fold_best_c = min(c_losses, key=lambda c: (round(c_losses[c], 8), c))

        fold_c_selected.append(fold_best_c)

        # Fit model on training fold and calibrate on inner calibration holdout
        base_pipe = build_outsider_logreg_pipeline(fold_best_c)
        if base_idx is not None and cal_idx is not None and len(cal_idx) > 0:
            base_pipe.fit(
                X_train.iloc[base_idx],
                y_train.iloc[base_idx],
                model__sample_weight=train_weights[base_idx],
            )
            try:
                if HAS_FROZEN_ESTIMATOR and FrozenEstimator is not None:
                    calibrated = CalibratedClassifierCV(
                        estimator=FrozenEstimator(base_pipe),
                        method="sigmoid",
                        cv=None,
                    )
                else:
                    calibrated = CalibratedClassifierCV(
                        estimator=base_pipe,
                        method="sigmoid",
                        cv="prefit",
                    )
                calibrated.fit(
                    X_train.iloc[cal_idx],
                    y_train.iloc[cal_idx],
                    sample_weight=train_weights[cal_idx],
                )
                val_probs = calibrated.predict_proba(X_val)[:, 1]
                fold_models.append(calibrated)
            except Exception:
                val_probs = base_pipe.predict_proba(X_val)[:, 1]
                fold_models.append(base_pipe)
        else:
            base_pipe.fit(X_train, y_train, model__sample_weight=train_weights)
            val_probs = base_pipe.predict_proba(X_val)[:, 1]
            fold_models.append(base_pipe)

        oof_probs[val_idx] = val_probs

    # Item 1.18: DO NOT fill unpredicted rows with fallback_mid!
    # Early warmup blocks intentionally lack OOF; missing OOF must remain NaN and not mimic M0.

    # Overall evaluation metrics
    ask_vals = df["executable_ask"].to_numpy() if "executable_ask" in df.columns else None
    eval_metrics = evaluate_outsider_predictions(
        y_true=y.to_numpy(),
        p_win=oof_probs,
        executable_ask=ask_vals,
        fee_rate=fee_rate,
        min_edge=min_edge,
    )

    all_weights = market_balanced_weights(groups)
    try:
        f_base_idx, f_cal_idx = _group_holdout_indices(X, y, groups, None, validation_fraction=0.20)
        if len(np.unique(y.iloc[f_base_idx])) < 2:
            f_base_idx = np.arange(len(X))
            f_cal_idx = None
    except Exception:
        f_base_idx = np.arange(len(X))
        f_cal_idx = None

    # Item 1.16: Final C selected on development holdout, NOT by median of folds
    best_overall_c = 1.0
    if f_base_idx is not None and f_cal_idx is not None and len(f_cal_idx) > 0:
        dev_c_losses = {}
        for c_cand in C_GRID:
            cand_pipe = build_outsider_logreg_pipeline(c_cand)
            try:
                cand_pipe.fit(X.iloc[f_base_idx], y.iloc[f_base_idx], model__sample_weight=all_weights[f_base_idx])
                dev_p = cand_pipe.predict_proba(X.iloc[f_cal_idx])[:, 1]
                dev_c_losses[c_cand] = log_loss_score(y.iloc[f_cal_idx], dev_p, sample_weight=all_weights[f_cal_idx])
            except Exception:
                pass
        if dev_c_losses:
            best_overall_c = min(dev_c_losses, key=lambda c: (round(dev_c_losses[c], 8), c))
    elif fold_c_selected:
        best_overall_c = fold_c_selected[0]

    # Fit final model on all data
    final_pipe = build_outsider_logreg_pipeline(best_overall_c)
    try:
        if f_base_idx is not None and f_cal_idx is not None and len(f_cal_idx) > 0:
            final_pipe.fit(X.iloc[f_base_idx], y.iloc[f_base_idx], model__sample_weight=all_weights[f_base_idx])
            final_calib = CalibratedClassifierCV(
                estimator=FrozenEstimator(final_pipe) if HAS_FROZEN_ESTIMATOR and FrozenEstimator else final_pipe,
                method="sigmoid",
                cv=None if HAS_FROZEN_ESTIMATOR and FrozenEstimator else "prefit",
            )
            final_calib.fit(X.iloc[f_cal_idx], y.iloc[f_cal_idx], sample_weight=all_weights[f_cal_idx])
            final_model = final_calib
        else:
            final_pipe.fit(X, y, model__sample_weight=all_weights)
            final_model = final_pipe
    except Exception:
        final_pipe.fit(X, y, model__sample_weight=all_weights)
        final_model = final_pipe

    return OutsiderTrainResult(
        feature_set=feature_set,
        feature_names=feats,
        best_c=best_overall_c,
        oof_predictions=oof_probs,
        metrics=eval_metrics,
        fold_models=fold_models,
        final_model=final_model,
        c_selected_per_fold=fold_c_selected,
    )


def compute_paired_model_deltas(
    res_a: OutsiderTrainResult,
    res_b: OutsiderTrainResult,
) -> dict[str, float]:
    """
    Computes paired deltas between Model A and Model B evaluated on the exact same dataset:
      delta_brier = Brier_B - Brier_A (negative is better)
      delta_log_loss = LogLoss_B - LogLoss_A (negative is better)
      delta_ece = ECE_B - ECE_A (negative is better)
      delta_pnl = PnL_B - PnL_A (positive is better)
      delta_expectancy = Expectancy_B - Expectancy_A (positive is better)
    """
    m_a = res_a.metrics
    m_b = res_b.metrics

    return {
        "delta_brier": round(float(m_b["brier"] - m_a["brier"]), 6),
        "delta_log_loss": round(float(m_b["log_loss"] - m_a["log_loss"]), 6),
        "delta_ece": round(float(m_b["ece"] - m_a["ece"]), 6),
        "delta_pnl": round(float(m_b["total_pnl"] - m_a["total_pnl"]), 4),
        "delta_expectancy": round(float(m_b["expectancy"] - m_a["expectancy"]), 6),
        "brier_a": m_a["brier"],
        "brier_b": m_b["brier"],
        "log_loss_a": m_a["log_loss"],
        "log_loss_b": m_b["log_loss"],
        "ece_a": m_a["ece"],
        "ece_b": m_b["ece"],
        "pnl_a": m_a["total_pnl"],
        "pnl_b": m_b["total_pnl"],
    }
