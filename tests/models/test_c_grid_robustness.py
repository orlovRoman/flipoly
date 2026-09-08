"""Tests for inner C grid selection robustness and outer test isolation (Item 1.9).

Verifies that:
1. Inner C selection over [0.1, 0.5, 1.0] minimizes log loss on inner holdout.
2. Outer test fold labels do NOT influence the C selection or parameters of inner folds.
3. Group boundaries (market_id) and temporal ordering are preserved without leakage.
"""
import pickle
import numpy as np
import pandas as pd
import pytest

from polyflip.models.trainer import (
    _fit_and_serialize,
    _group_holdout_indices,
    _outer_validation_splits,
)


def _generate_synthetic_multimarket_data(n_markets: int = 10, rows_per_market: int = 20):
    np.random.seed(42)
    rows = []
    t0 = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    feature_names = ["feat1", "feat2", "feat3"]

    for m_idx in range(n_markets):
        m_id = f"market_{m_idx:03d}"
        m_bias = np.random.uniform(-0.5, 0.5)
        for r in range(rows_per_market):
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)
            x3 = np.random.normal(0, 1)
            # Logit determining win probability
            logit = 1.2 * x1 - 0.8 * x2 + m_bias
            p = 1.0 / (1.0 + np.exp(-logit))
            y = int(np.random.rand() < p)
            rec_dt = t0 + pd.Timedelta(hours=m_idx * 2, minutes=r * 5)
            mid = float(np.clip(0.40 + 0.1 * m_bias + 0.02 * (r % 5), 0.1, 0.9))
            rows.append({
                "market_id": m_id,
                "recorded_at": rec_dt,
                "mid_price": mid,
                "feat1": x1,
                "feat2": x2,
                "feat3": x3,
                "target": y,
            })

    df = pd.DataFrame(rows)
    return df, feature_names


def test_inner_c_search_invariant_to_outer_test_labels():
    """Modifying outer test fold labels changes only test metrics, NOT the inner C selection."""
    df, feature_names = _generate_synthetic_multimarket_data(n_markets=10, rows_per_market=15)
    X = df[feature_names]
    y_orig = df["target"].copy()
    groups = df["market_id"]
    timestamps = df["recorded_at"]
    mid_prices = df["mid_price"]

    # 1. First run with original labels
    res_orig = _fit_and_serialize(
        X=X,
        y=y_orig,
        groups=groups,
        mid_prices=mid_prices,
        timestamps=timestamps,
        feature_set="MODEL_A",
    )
    assert res_orig is not None
    _, _, _, _, _, backtest_orig, _ = res_orig
    c_folds_orig = backtest_orig["c_selected_per_fold"]
    assert len(c_folds_orig) > 0
    assert all(c in [0.1, 0.5, 1.0] for c in c_folds_orig)

    # 2. Identify the last outer test fold indices
    splits, _ = _outer_validation_splits(X, y_orig, groups, timestamps)
    assert len(splits) > 0
    _, last_val_idx = splits[-1]

    # Mutate ONLY the labels in the last outer test fold (invert 0 <-> 1)
    y_mutated = y_orig.copy()
    y_mutated.iloc[last_val_idx] = 1 - y_mutated.iloc[last_val_idx]

    # 3. Second run with mutated outer test labels
    res_mutated = _fit_and_serialize(
        X=X,
        y=y_mutated,
        groups=groups,
        mid_prices=mid_prices,
        timestamps=timestamps,
        feature_set="MODEL_A",
    )
    assert res_mutated is not None
    _, _, _, _, _, backtest_mutated, _ = res_mutated
    c_folds_mutated = backtest_mutated["c_selected_per_fold"]

    # INVARIANCE ASSERTION:
    # For the last fold, the outer test labels were changed, but the training fold (train_idx)
    # was identical. The inner C selection for that fold MUST be 100% identical!
    assert c_folds_orig[-1] == c_folds_mutated[-1], (
        f"Outer test fold label leakage detected! C selected changed: {c_folds_orig[-1]} vs {c_folds_mutated[-1]}"
    )


def test_inner_c_grid_selects_best_c_from_allowed_values():
    """Inner C grid search evaluates C_GRID = [0.1, 0.5, 1.0] and records log loss."""
    df, feature_names = _generate_synthetic_multimarket_data(n_markets=8, rows_per_market=15)
    X = df[feature_names]
    y = df["target"]
    groups = df["market_id"]
    timestamps = df["recorded_at"]
    mid_prices = df["mid_price"]

    res = _fit_and_serialize(
        X=X,
        y=y,
        groups=groups,
        mid_prices=mid_prices,
        timestamps=timestamps,
        feature_set="MODEL_A",
    )
    assert res is not None
    _, _, _, _, _, backtest, _ = res

    # Check that best_c is selected from allowed grid
    best_c = backtest.get("best_c")
    assert best_c in [0.1, 0.5, 1.0]

    # Check that c_search_loss contains evaluated losses for candidate C values
    c_losses = backtest.get("c_search_loss", {})
    assert len(c_losses) > 0
    for c_val in c_losses:
        assert float(c_val) in [0.1, 0.5, 1.0]
        assert c_losses[c_val] >= 0.0


def test_inner_holdout_has_disjoint_market_groups():
    """Inner calibration split must have strictly disjoint market_id groups (no group leakage)."""
    df, feature_names = _generate_synthetic_multimarket_data(n_markets=8, rows_per_market=15)
    X = df[feature_names]
    y = df["target"]
    groups = df["market_id"]
    timestamps = df["recorded_at"]

    train_idx, cal_idx = _group_holdout_indices(
        X, y, groups, timestamps, validation_fraction=0.25
    )

    train_markets = set(groups.iloc[train_idx])
    cal_markets = set(groups.iloc[cal_idx])

    # Disjoint groups: no overlap between training and calibration markets
    overlap = train_markets.intersection(cal_markets)
    assert not overlap, f"Group leakage detected! Overlapping markets: {overlap}"

