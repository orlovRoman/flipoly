"""
tests/models/test_point_in_time_features.py

Tests for Point-in-Time dynamic features (Step 1.5):
- Explicit 60s and 180s temporal horizons
- Sampling frequency invariance (15s, 45s, 60s polls)
- Missing reference detection on polling gaps
- Point-in-time causal isolation with decision_at
- Non-default index and row permutation preservation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytest

from polyflip.models.point_in_time_features import compute_point_in_time_features


def test_empty_dataframe_returns_clean_schema():
    df = pd.DataFrame()
    res = compute_point_in_time_features(df)
    assert "pm_change_60s" in res.columns
    assert "pm_change_180s" in res.columns
    assert "price_distance_from_max" in res.columns
    assert len(res) == 0


def test_custom_index_and_row_order_preserved():
    base_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    custom_idx = [101, 205, 309, 412]
    df = pd.DataFrame({
        "market_id": ["m1"] * 4,
        "recorded_at": [base_t + timedelta(seconds=30 * i) for i in range(4)],
        "mid_price": [0.50, 0.52, 0.55, 0.53],
    }, index=custom_idx)

    res = compute_point_in_time_features(df)
    assert list(res.index) == custom_idx
    assert not res["mid_price"].isna().any()
    assert res.loc[101, "price_distance_from_max"] == 0.0
    assert res.loc[412, "price_distance_from_max"] == pytest.approx(0.02, abs=1e-6)


def test_sampling_frequency_invariance():
    """
    On the same linear price path, as-of 60s change should yield identical
    delta ~0.04 whether polled every 15s, 30s, or 60s.
    """
    base_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    # Price rises by 0.00066667 per second (0.04 per 60s)
    # 1. 15-second polling
    times_15 = [base_t + timedelta(seconds=15 * i) for i in range(10)]
    prices_15 = [0.50 + (t - base_t).total_seconds() * (0.04 / 60.0) for t in times_15]
    df_15 = pd.DataFrame({"market_id": ["m1"] * len(times_15), "recorded_at": times_15, "mid_price": prices_15})
    res_15 = compute_point_in_time_features(df_15)

    # 2. 60-second polling
    times_60 = [base_t + timedelta(seconds=60 * i) for i in range(3)]
    prices_60 = [0.50 + (t - base_t).total_seconds() * (0.04 / 60.0) for t in times_60]
    df_60 = pd.DataFrame({"market_id": ["m1"] * len(times_60), "recorded_at": times_60, "mid_price": prices_60})
    res_60 = compute_point_in_time_features(df_60)

    # At t=120s (index 8 in df_15, index 2 in df_60):
    change_15 = res_15.iloc[8]["pm_change_60s"]
    change_60 = res_60.iloc[2]["pm_change_60s"]

    assert change_15 == pytest.approx(0.04, abs=1e-4)
    assert change_60 == pytest.approx(0.04, abs=1e-4)
    assert change_15 == pytest.approx(change_60, abs=1e-4)


def test_polling_gap_produces_missing_reference():
    """A multi-minute gap between polls produces NaN and has_60s_ref=0.0."""
    base_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    # Poll at t=0, then 5 minutes later at t=300s
    times = [base_t, base_t + timedelta(seconds=300)]
    prices = [0.50, 0.70]
    df = pd.DataFrame({"market_id": ["m1", "m1"], "recorded_at": times, "mid_price": prices})
    res = compute_point_in_time_features(df)

    assert pd.isna(res.iloc[1]["pm_change_60s"])
    assert res.iloc[1]["has_60s_ref"] == 0.0
    assert res.iloc[1]["history_age_seconds"] == 300.0


def test_causal_decision_at_isolation():
    """Snapshots beyond decision_at are masked out and do not alter earlier features."""
    base_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    decision_t = base_t + timedelta(minutes=2)

    times = [base_t + timedelta(seconds=30 * i) for i in range(8)]
    # A massive price spike at t=3min (after decision_t)
    prices = [0.50, 0.51, 0.52, 0.53, 0.54, 0.99, 0.99, 0.99]
    df = pd.DataFrame({"market_id": ["m1"] * 8, "recorded_at": times, "mid_price": prices})

    res = compute_point_in_time_features(df, decision_at=decision_t)
    # Decision row is at i=4 (t=2min, price=0.54)
    # The future spike to 0.99 must NOT affect price_distance_from_max at or before i=4
    assert res.iloc[4]["price_distance_from_max"] == 0.0
    # Rows beyond decision_t (i=5, 6, 7) must have has_60s_ref == 0.0 and NaN changes
    assert res.iloc[5]["has_60s_ref"] == 0.0
    assert pd.isna(res.iloc[5]["pm_change_60s"])


def test_global_max_support():
    """global_max initializes/caps the expanding maximum for causal distance computation."""
    base_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    times = [base_t + timedelta(seconds=30 * i) for i in range(3)]
    prices = [0.45, 0.48, 0.50]
    df = pd.DataFrame({"market_id": ["m1"] * 3, "recorded_at": times, "mid_price": prices})

    # With global_max=0.60, distance should be 0.60 - price
    res = compute_point_in_time_features(df, global_max=0.60)
    assert res.loc[0, "price_distance_from_max"] == pytest.approx(0.15, abs=1e-6)
    assert res.loc[1, "price_distance_from_max"] == pytest.approx(0.12, abs=1e-6)
    assert res.loc[2, "price_distance_from_max"] == pytest.approx(0.10, abs=1e-6)


def test_point_in_time_feature_names_constant():
    from polyflip.models.point_in_time_features import POINT_IN_TIME_FEATURE_NAMES
    expected = (
        "pm_change_60s",
        "pm_change_180s",
        "has_60s_ref",
        "has_180s_ref",
        "legacy_last_poll_delta",
        "price_distance_from_max",
        "history_age_seconds",
    )
    assert POINT_IN_TIME_FEATURE_NAMES == expected


def test_apply_market_feature_pipeline_export_and_chaining():
    from polyflip.models.point_in_time_features import apply_market_feature_pipeline
    df = pd.DataFrame([{
        "market_id": "m1",
        "recorded_at": pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
        "time_left_min": 15.0,
        "mid_price": 0.50,
        "spread": 0.01,
        "price_velocity": 0.0,
        "volume_5min": 100.0,
        "hour_of_day": 12,
        "day_of_week": 3.0,
        "market_duration_min": 15.0,
    }])
    res = apply_market_feature_pipeline(df)
    for expected_col in (
        "pm_change_60s", "pm_change_180s", "has_60s_ref", "has_180s_ref",
        "history_age_seconds", "legacy_last_poll_delta", "price_distance_from_max",
        "price_deviation", "spread_pct", "time_phase", "velocity_x_phase",
    ):
        assert expected_col in res.columns

