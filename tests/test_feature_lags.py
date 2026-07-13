import pandas as pd
from datetime import datetime, timezone
from polyflip.models.trainer import add_derived_features, DERIVED_FEATURES
from polyflip.models.feature_lags import add_lag_features, LAG_FEATURE_NAMES
from polyflip.trading.feature_builder import FEATURE_COLUMNS

def test_day_of_week_range():
    for weekday in range(7):
        dt = datetime(2026, 7, 13 + weekday, tzinfo=timezone.utc)
        assert dt.weekday() == weekday
        assert 0 <= dt.weekday() <= 6

def test_price_distance_from_max():
    df = pd.DataFrame({
        "market_id": ["m1", "m1", "m1"],
        "mid_price":  [0.6, 0.8, 0.7],
        "spread": [0.02]*3, 
        "time_left_min": [60, 30, 10],
    })
    r = add_derived_features(df)
    assert abs(r.loc[0, "price_distance_from_max"] - 0.2) < 1e-6  # 0.8 - 0.6
    assert abs(r.loc[1, "price_distance_from_max"] - 0.0) < 1e-6  # max == self
    assert (r["price_distance_from_max"] >= 0).all()

def test_feature_columns_match_derived_plus_lag():
    all_expected = set(DERIVED_FEATURES) | set(LAG_FEATURE_NAMES) | {
        "time_left_min", "mid_price", "spread",
        "volume_5min", "price_velocity", "hour_of_day",
        "day_of_week", "price_distance_from_max"
    }
    assert set(FEATURE_COLUMNS) == all_expected, (
        f"Расхождение: {set(FEATURE_COLUMNS).symmetric_difference(all_expected)}"
    )

def test_add_lag_features_basic():
    df = pd.DataFrame({
        "market_id": ["m1", "m1", "m1", "m1", "m1", "m1", "m1", "m1"],
        "recorded_at": pd.date_range("2026-07-13", periods=8, freq="5min"),
        "mid_price": [0.5, 0.52, 0.51, 0.55, 0.53, 0.54, 0.58, 0.59],
        "spread": [0.01]*8,
        "volume_5min": [100, 150, 120, 200, 210, 180, 250, 300],
        "price_velocity": [0.0, 0.02, -0.01, 0.04, -0.02, 0.01, 0.04, 0.01]
    })
    
    r = add_lag_features(df)
    
    # price_velocity_lag1: shift(1)
    assert r.loc[1, "price_velocity_lag1"] == 0.0
    assert r.loc[2, "price_velocity_lag1"] == 0.02
    
    # price_momentum: shift(3)
    # i=3: mid_price=0.55, lag3 (i=0) = 0.50 => 0.05
    assert abs(r.loc[3, "price_momentum"] - 0.05) < 1e-6
    
    # volume_trend: shift(3)
    # i=3: vol=200, lag3=100 => 200/100 = 2.0
    assert abs(r.loc[3, "volume_trend"] - 2.0) < 1e-6

    # spread_trend: shift(6)
    # i=6: spread=0.01, lag6(i=0)=0.01 => 1.0
    assert abs(r.loc[6, "spread_trend"] - 1.0) < 1e-6
