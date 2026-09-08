"""
tests/models/test_outsider_features.py

Unit tests for normalized strike distance (z_outsider, Item 2.2)
and short directional momentum (ret_outsider_30s, ret_outsider_120s, Item 2.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polyflip.models.point_in_time_features import (
    compute_normalized_strike_distance,
    compute_directional_momentum,
    compute_outsider_model_features,
)
from polyflip.models.outsider_feature_sets import (
    get_outsider_feature_set,
    MODEL_A1_FEATURES,
    MODEL_B1_FEATURES,
)


def test_outsider_feature_sets_contracts_v2():
    """Verify contracts for Model A1 and Model B1."""
    set_a1 = get_outsider_feature_set("MODEL_A1")
    assert set_a1.features == MODEL_A1_FEATURES
    assert set_a1.version == "model-a1-compact-v2"
    assert len(set_a1.schema_hash) == 16

    set_b1 = get_outsider_feature_set("MODEL_B1")
    assert set_b1.features == MODEL_B1_FEATURES
    assert set_b1.version == "model-b1-momentum-v1"
    assert len(set_b1.schema_hash) == 16

    # Model B alias points to Model B1
    set_b = get_outsider_feature_set("MODEL_B")
    assert set_b.features == set_b1.features


def test_strike_distance_zero_at_strike():
    """Item 2.2: When S == K, z_outsider must be exactly 0.0."""
    z_up, ok_up = compute_normalized_strike_distance(
        underlying_price=50000.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=5.0,
        candidate_side="UP",
    )
    assert ok_up is True
    assert z_up == pytest.approx(0.0, abs=1e-12)

    z_down, ok_down = compute_normalized_strike_distance(
        underlying_price=50000.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=5.0,
        candidate_side="DOWN",
    )
    assert ok_down is True
    assert z_down == pytest.approx(0.0, abs=1e-12)


def test_strike_distance_side_inversion_flips_sign():
    """Item 2.2: Switching candidate side UP <-> DOWN inverts the sign: z_DOWN == -z_UP."""
    z_up, ok_up = compute_normalized_strike_distance(
        underlying_price=50500.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=5.0,
        candidate_side="UP",
    )
    z_down, ok_down = compute_normalized_strike_distance(
        underlying_price=50500.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=5.0,
        candidate_side="DOWN",
    )
    assert ok_up is True and ok_down is True
    assert z_up > 0.0  # Above strike is favorable for UP candidate
    assert z_down < 0.0  # Above strike is unfavorable for DOWN candidate
    assert z_down == pytest.approx(-z_up, rel=1e-9)


def test_strike_distance_tau_and_sigma_scaling():
    """Item 2.2: Increasing tau or sigma increases denominator and strictly decreases |z|."""
    z_base, _ = compute_normalized_strike_distance(
        underlying_price=51000.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=2.0,
        candidate_side="UP",
    )

    # Longer time left (tau = 8 min vs 2 min): |z| must decrease
    z_longer_tau, _ = compute_normalized_strike_distance(
        underlying_price=51000.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=8.0,
        candidate_side="UP",
    )
    assert abs(z_longer_tau) < abs(z_base)
    # Factor is exactly sqrt(2/8) = 0.5
    assert z_longer_tau == pytest.approx(z_base * 0.5, rel=1e-6)

    # Higher volatility (sigma = 0.004 vs 0.002): |z| must decrease
    z_higher_vol, _ = compute_normalized_strike_distance(
        underlying_price=51000.0,
        strike_price=50000.0,
        sigma_1m=0.004,
        tau_min=2.0,
        candidate_side="UP",
    )
    assert abs(z_higher_vol) < abs(z_base)
    assert z_higher_vol == pytest.approx(z_base * 0.5, rel=1e-6)


def test_strike_distance_tau_zero_and_missing_volatility_safety():
    """Item 2.2: Zero tau or invalid volatility handled gracefully without zero division."""
    # tau = 0
    z_zero_tau, ok_zero_tau = compute_normalized_strike_distance(
        underlying_price=50500.0,
        strike_price=50000.0,
        sigma_1m=0.002,
        tau_min=0.0,
        candidate_side="UP",
    )
    assert np.isfinite(z_zero_tau)
    assert ok_zero_tau is True

    # Zero volatility
    z_zero_vol, ok_zero_vol = compute_normalized_strike_distance(
        underlying_price=50500.0,
        strike_price=50000.0,
        sigma_1m=0.0,
        tau_min=5.0,
        candidate_side="UP",
    )
    assert ok_zero_vol is False
    assert z_zero_vol == 0.0

    # Negative/NaN strike
    z_nan_k, ok_nan_k = compute_normalized_strike_distance(
        underlying_price=50500.0,
        strike_price=np.nan,
        sigma_1m=0.002,
        tau_min=5.0,
        candidate_side="UP",
    )
    assert ok_nan_k is False
    assert z_nan_k == 0.0


def test_directional_momentum_positive_towards_winning_direction():
    """Item 2.3: Movement towards outsider win gives positive sign for both UP and DOWN."""
    # 1. Candidate UP, underlying rises (50000 -> 50500): positive return
    ret_up_win, ok1 = compute_directional_momentum(
        underlying_current=50500.0,
        underlying_lagged=50000.0,
        candidate_side="UP",
    )
    assert ok1 is True
    assert ret_up_win > 0.0
    assert ret_up_win == pytest.approx(float(np.log(50500.0 / 50000.0)), rel=1e-9)

    # 2. Candidate DOWN, underlying drops (50500 -> 50000): positive return
    ret_down_win, ok2 = compute_directional_momentum(
        underlying_current=50000.0,
        underlying_lagged=50500.0,
        candidate_side="DOWN",
    )
    assert ok2 is True
    assert ret_down_win > 0.0
    assert ret_down_win == pytest.approx(float(np.log(50500.0 / 50000.0)), rel=1e-9)

    # 3. Candidate UP, underlying drops: negative return
    ret_up_lose, ok3 = compute_directional_momentum(
        underlying_current=50000.0,
        underlying_lagged=50500.0,
        candidate_side="UP",
    )
    assert ok3 is True
    assert ret_up_lose < 0.0

    # 4. Candidate DOWN, underlying rises: negative return
    ret_down_lose, ok4 = compute_directional_momentum(
        underlying_current=50500.0,
        underlying_lagged=50000.0,
        candidate_side="DOWN",
    )
    assert ok4 is True
    assert ret_down_lose < 0.0


def test_directional_momentum_missing_reference():
    """Item 2.3: When lagged observation is unavailable, has_ref must be False and ret=0.0."""
    ret_miss, ok_miss = compute_directional_momentum(
        underlying_current=50000.0,
        underlying_lagged=np.nan,
        candidate_side="UP",
        has_ref=False,
    )
    assert ok_miss is False
    assert ret_miss == 0.0


def test_compute_outsider_model_features_pipeline():
    """Verify complete feature computation pipeline for Model A1 and Model B1."""
    df_raw = pd.DataFrame({
        "market_id": ["m1", "m1", "m2"],
        "mid_price": [0.35, 0.40, 0.25],
        "spread": [0.02, 0.03, 0.01],
        "time_left_min": [10.0, 5.0, 2.0],
        "candidate_side": ["UP", "UP", "DOWN"],
        "underlying_price": [50200.0, 50400.0, 49800.0],
        "strike_value": [50000.0, 50000.0, 50000.0],
        "underlying_lag_30s": [50150.0, 50350.0, 49900.0],
        "underlying_lag_120s": [50100.0, 50300.0, 50000.0],
    })

    df_feats = compute_outsider_model_features(df_raw)

    # Check Model A1 features present
    for f in MODEL_A1_FEATURES:
        assert f in df_feats.columns
        assert df_feats[f].isna().sum() == 0

    # Check Model B1 features present
    for f in MODEL_B1_FEATURES:
        assert f in df_feats.columns
        assert df_feats[f].isna().sum() == 0

    assert "has_z_ref" in df_feats.columns
    assert "has_ret_30s_ref" in df_feats.columns
    assert "has_ret_120s_ref" in df_feats.columns
    assert df_feats["has_z_ref"].all()
    assert df_feats["has_ret_30s_ref"].all()
    assert df_feats["has_ret_120s_ref"].all()
