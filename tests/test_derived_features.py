import numpy as np
import pandas as pd
import pytest
from polyflip.models.trainer import add_derived_features, DERIVED_FEATURES

def test_values():
    df = pd.DataFrame([
        {"mid_price": 0.5, "time_left_min": 7.0,  "spread": 0.02},
        {"mid_price": 0.9, "time_left_min": 2.0,  "spread": 0.01},
        {"mid_price": 0.1, "time_left_min": 14.0, "spread": 0.05},
    ])
    r = add_derived_features(df)

    assert r.loc[0, "price_deviation"]    == pytest.approx(0.0)
    assert r.loc[1, "price_deviation"]    == pytest.approx(0.4)
    assert r.loc[1, "deviation_x_time"]   == pytest.approx(0.4 * 2.0)   # 0.8
    assert r.loc[2, "deviation_x_time"]   == pytest.approx(0.4 * 14.0)  # 5.6
    assert r.loc[1, "price_deviation_sq"] == pytest.approx(0.16)
    assert r.loc[1, "spread_pct"]         == pytest.approx(0.01 / 0.9, rel=1e-3)
    assert r.loc[0, "log_time_left"]      == pytest.approx(np.log1p(7.0))

def test_no_mutation():
    df = pd.DataFrame([{"mid_price": 0.7, "time_left_min": 5.0, "spread": 0.03}])
    add_derived_features(df)
    assert "price_deviation" not in df.columns

def test_all_derived_columns_present():
    from polyflip.models.feature_lags import LAG_FEATURE_NAMES
    df = pd.DataFrame([{"mid_price": 0.6, "time_left_min": 8.0, "spread": 0.02}])
    r = add_derived_features(df)
    for feat in DERIVED_FEATURES:
        if feat in LAG_FEATURE_NAMES:
            continue
        assert feat in r.columns

def test_spread_pct_near_zero_price():
    df = pd.DataFrame([{"mid_price": 0.0, "time_left_min": 5.0, "spread": 0.01}])
    r = add_derived_features(df)
    # Не должно быть inf или NaN
    assert np.isfinite(r.loc[0, "spread_pct"])
    # При mid_price=0: 0.01 / 1e-6 = 10000 -> клиппинг до 10.0
    assert r.loc[0, "spread_pct"] == pytest.approx(10.0)
