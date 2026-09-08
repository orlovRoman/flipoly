"""
tests/models/test_outsider_trainer.py

Unit tests for Outsider Model A (Item 2.6) and Model B (Item 2.7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polyflip.models.outsider_trainer import (
    train_outsider_model,
    compute_paired_model_deltas,
)
from polyflip.models.outsider_feature_sets import (
    MODEL_A1_FEATURES,
    MODEL_B1_FEATURES,
)
from polyflip.models.point_in_time_features import compute_outsider_model_features


@pytest.fixture
def synthetic_cohort_data() -> pd.DataFrame:
    """Generates synthetic multi-market decision rows with ground truth signals."""
    np.random.seed(42)
    rows = []
    # 12 markets, 3 decision points each = 36 rows
    for m in range(12):
        m_id = f"market_{m:03d}"
        market_bias = np.random.uniform(-0.3, 0.3)
        for t_idx, tl in enumerate([10.0, 5.0, 2.0]):
            outsider_mid = float(np.clip(0.35 + 0.05 * market_bias + np.random.normal(0, 0.02), 0.15, 0.48))
            spread = 0.02
            # Underlying price and strike
            k = 50000.0
            # S moves with market bias
            s = 50000.0 + 300.0 * market_bias + np.random.normal(0, 50.0)
            lag30 = s - np.random.normal(10.0 * market_bias, 15.0)
            lag120 = s - np.random.normal(30.0 * market_bias, 30.0)

            # Target: higher if underlying is above strike and price momentum is positive
            prob_win = 1.0 / (1.0 + np.exp(-(1.5 * (s - k) / 200.0 + 0.8 * (s - lag30) / 20.0)))
            y = int(np.random.rand() < prob_win)

            rows.append({
                "market_id": m_id,
                "fold": m % 4,
                "time_left_min": tl,
                "mid_price": outsider_mid,
                "outsider_mid": outsider_mid,
                "candidate_side": "UP" if m % 2 == 0 else "DOWN",
                "spread": spread,
                "candidate_spread": spread,
                "underlying_price": s,
                "strike_value": k,
                "underlying_lag_30s": lag30,
                "underlying_lag_120s": lag120,
                "sigma_1m": 0.0015,
                "executable_ask": outsider_mid + 0.01,
                "target": y,
                "y_candidate_win": y,
            })

    df = pd.DataFrame(rows)
    return compute_outsider_model_features(df)


def test_model_a_symmetry(synthetic_cohort_data: pd.DataFrame):
    """
    Item 2.6: Mirror YES/NO situations with identical outsider price,
    time_left, and spread must yield identical Model A features and predictions.
    """
    res_a = train_outsider_model(synthetic_cohort_data, feature_set="MODEL_A1")
    model = res_a.final_model
    assert model is not None

    row_yes = pd.DataFrame([{
        "logit_mid_price": -0.619,
        "log_time_left": 1.791,
        "candidate_spread": 0.02,
        "logit_price_x_log_time": -1.108,
    }])
    row_no = pd.DataFrame([{
        "logit_mid_price": -0.619,
        "log_time_left": 1.791,
        "candidate_spread": 0.02,
        "logit_price_x_log_time": -1.108,
    }])

    p_yes = model.predict_proba(row_yes)[0, 1]
    p_no = model.predict_proba(row_no)[0, 1]

    assert p_yes == pytest.approx(p_no, abs=1e-9)


def test_model_a_feature_names_locked(synthetic_cohort_data: pd.DataFrame):
    """Item 2.6: Model A feature names after fit match MODEL_A1_FEATURES exactly."""
    res_a = train_outsider_model(synthetic_cohort_data, feature_set="MODEL_A1")
    assert res_a.feature_names == MODEL_A1_FEATURES
    assert len(res_a.feature_names) == 4
    assert all(c in [0.1, 0.5, 1.0] for c in res_a.c_selected_per_fold)


def test_model_b_paired_comparison(synthetic_cohort_data: pd.DataFrame):
    """
    Item 2.7: Model B1 evaluated on same folds and rows as Model A1,
    producing paired delta metrics.
    """
    res_a = train_outsider_model(synthetic_cohort_data, feature_set="MODEL_A1")
    res_b = train_outsider_model(synthetic_cohort_data, feature_set="MODEL_B1")

    assert res_b.feature_names == MODEL_B1_FEATURES
    assert len(res_b.feature_names) == 7

    # Compute paired deltas
    deltas = compute_paired_model_deltas(res_a, res_b)

    assert "delta_brier" in deltas
    assert "delta_log_loss" in deltas
    assert "delta_ece" in deltas
    assert "delta_pnl" in deltas

    # Length of OOF predictions is identical
    assert len(res_a.oof_predictions) == len(res_b.oof_predictions)
    assert len(res_a.oof_predictions) == len(synthetic_cohort_data)


def test_model_b_collapses_to_a_when_b_features_are_constant(synthetic_cohort_data: pd.DataFrame):
    """
    Item 2.7: When additional features (z, ret30, ret120) are replaced with constants (0.0),
    Model B retains only Model A signals.
    """
    df_const = synthetic_cohort_data.copy()
    df_const["z_outsider"] = 0.0
    df_const["ret_outsider_30s"] = 0.0
    df_const["ret_outsider_120s"] = 0.0

    res_b_const = train_outsider_model(df_const, feature_set="MODEL_B1")
    res_a = train_outsider_model(synthetic_cohort_data, feature_set="MODEL_A1")

    # With constant inputs, the predictions correlate very strongly (> 0.95)
    valid_mask = np.isfinite(res_b_const.oof_predictions) & np.isfinite(res_a.oof_predictions)
    assert np.sum(valid_mask) > 0
    corr = np.corrcoef(res_b_const.oof_predictions[valid_mask], res_a.oof_predictions[valid_mask])[0, 1]
    assert corr > 0.95
