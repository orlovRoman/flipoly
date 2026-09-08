import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from polyflip.models.probability_metrics import (
    brier_score,
    log_loss_score,
    expected_calibration_error,
    reliability_table,
)
from polyflip.models.trainer import _fit_and_serialize
from polyflip.models.temporal_validation import market_balanced_weights


def test_canonical_probability_metrics_values():
    """1.15: Verify canonical Brier score, log loss, and ECE calculation."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4, 0.85, 0.15])

    brier = brier_score(y_true, y_prob)
    assert 0.0 <= brier <= 1.0

    ll = log_loss_score(y_true, y_prob)
    assert ll > 0.0

    ece, diag = expected_calibration_error(y_true, y_prob, n_bins=5, min_samples=5)
    assert ece is not None
    assert 0.0 <= ece <= 1.0
    assert diag["status"] == "VALID"


def test_canonical_ece_no_fake_05_fallback():
    """1.15: For insufficient samples, ECE returns None rather than fabricating 0.5."""
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])

    ece, diag = expected_calibration_error(y_true, y_prob, min_samples=10)
    assert ece is None
    assert diag["status"] == "TOO_FEW_SAMPLES"


def test_market_snapshot_duplication_preserves_total_weight():
    """1.14: Duplicating snapshots for a market preserves total market weight and does not distort balance."""
    groups = pd.Series(["m1", "m1", "m2"])
    w1 = market_balanced_weights(groups)
    assert np.isclose(w1[groups == "m1"].sum(), w1[groups == "m2"].sum())

    # High polling frequency on m1 (20 snapshots) vs standard m2 (2 snapshots)
    groups_dense = pd.Series(["m1"] * 20 + ["m2"] * 2)
    w_dense = market_balanced_weights(groups_dense)
    assert np.isclose(w_dense[groups_dense == "m1"].sum(), w_dense[groups_dense == "m2"].sum())


def test_fit_and_serialize_inner_c_search_and_class_weight_none():
    """1.9 & 1.14: Inner C grid search using log loss and class_weight=None."""
    n_samples = 80
    np.random.seed(42)
    
    # Generate 10 markets with 8 snapshots each
    market_ids = []
    timestamps = []
    base_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    for m in range(10):
        for i in range(8):
            market_ids.append(f"m_{m}")
            timestamps.append(base_time + timedelta(hours=m, minutes=i * 2))

    X = pd.DataFrame({
        "mid_price": np.random.uniform(0.1, 0.9, n_samples),
        "time_left_min": np.random.uniform(2.0, 12.0, n_samples),
        "spread": np.random.uniform(0.01, 0.05, n_samples),
    })
    # Target correlated with mid_price
    y = pd.Series((X["mid_price"] > 0.5).astype(int))
    # Add a little noise to prevent perfect separation
    noise_idx = np.random.choice(n_samples, size=10, replace=False)
    y.iloc[noise_idx] = 1 - y.iloc[noise_idx]

    groups = pd.Series(market_ids)
    mid_prices = X["mid_price"]

    result = _fit_and_serialize(
        X=X,
        y=y,
        groups=groups,
        mid_prices=mid_prices,
        timestamps=pd.Series(timestamps),
        feature_set="MODEL_A",
    )

    assert result is not None
    model_bytes, val_acc, baseline_acc, optimal_threshold, ece, backtest, oof_artifact = result

    # Verify model has class_weight=None
    import pickle
    model = pickle.loads(model_bytes)
    # The base estimator inside CalibratedClassifierCV or pipeline
    if hasattr(model, "calibrated_classifiers_"):
        cal = model.calibrated_classifiers_[0]
        base_pipe = getattr(cal, "estimator", None)
    elif hasattr(model, "estimator"):
        base_pipe = getattr(model, "estimator")
    else:
        base_pipe = model

    lr_model = base_pipe.named_steps["model"]
    assert lr_model.class_weight is None, "Model class_weight must be None (harmonized weights)"

    # Verify SimpleImputer is in the pipeline
    assert "imputer" in base_pipe.named_steps

    # Verify backtest metadata records inner C search and canonical metrics
    assert "c_search_loss" in backtest
    assert "brier_score" in backtest
    assert "log_loss" in backtest
    assert "ece" in backtest
    assert backtest["model_config"]["class_weight"] is None
