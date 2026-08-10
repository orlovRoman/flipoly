import numpy as np
import pandas as pd
import pytest

from polyflip.models.trainer import _compute_backtest_pnl, _fit_and_serialize, add_derived_features
from polyflip.models.feature_lags import add_lag_features
from polyflip.trading.feature_builder import MarketSignal, build_feature_vector, FEATURE_COLUMNS


def test_flip_backtest_prices_yes_and_no_outsiders_symmetrically():
    result = _compute_backtest_pnl(
        oof_scores=np.array([0.9, 0.9]),
        y=pd.Series([1, 0]),
        mid_prices=pd.Series([0.8, 0.8]),
        threshold=0.5,
        fee_per_trade=0.02,
    )

    assert result["total_pnl"] == pytest.approx(0.56)
    assert result["strategy_branch"] == "OUTSIDER_ONLY"


def test_default_market_duration_is_fifteen_minutes():
    signal = MarketSignal(
        asset="BTC", mid_price=0.8, spread=0.01, volume_5min=1.0,
        price_velocity=0.0, hour_of_day=12, time_left_min=3.0,
    )
    vector = build_feature_vector(signal)
    final_phase_index = FEATURE_COLUMNS.index("is_final_phase")

    assert vector[0, final_phase_index] == 1.0


def test_derived_features_use_fifteen_minute_fallback():
    result = add_derived_features(pd.DataFrame({
        "mid_price": [0.8], "spread": [0.01], "time_left_min": [3.0],
    }))

    assert result.loc[0, "is_final_phase"] == 1.0


def test_legacy_derived_features_are_reconstructed_not_zeroed():
    result = add_derived_features(pd.DataFrame({
        "mid_price": [0.8], "spread": [0.01], "time_left_min": [3.0],
        "price_velocity": [0.02], "market_duration_min": [15.0],
    }))

    assert result.loc[0, "deviation_x_time"] == pytest.approx(0.9)
    assert result.loc[0, "price_deviation_sq"] == pytest.approx(0.09)
    assert result.loc[0, "time_phase"] == pytest.approx(0.2, abs=1e-6)
    assert result.loc[0, "velocity_x_phase"] == pytest.approx(0.016, abs=1e-6)
    assert result.loc[0, "dev_sq_x_phase"] == pytest.approx(0.072, abs=1e-6)


def test_legacy_features_are_not_permitted_zero_defaults():
    from polyflip.constants import ZERO_DEFAULT_FEATURES

    assert "price_deviation_sq" not in ZERO_DEFAULT_FEATURES

def test_calibrated_oof_path_defines_probabilities():
    rows = 100
    groups = pd.Series(np.repeat([f"market-{i}" for i in range(10)], 10))
    y = pd.Series(np.tile([0, 1], rows // 2))
    X = pd.DataFrame({
        "mid_price": np.where(y == 1, 0.4, 0.6),
        "time_left_min": np.tile(np.arange(1, 11), 10),
    })

    result = _fit_and_serialize(
        X, y, groups, mid_prices=X["mid_price"],
        min_precision=0.0, max_suspicious=1.0,
    )

    assert result is not None
    model_bytes, auc, baseline, threshold, ece, backtest = result
    assert model_bytes
    assert np.isfinite([auc, baseline, threshold, ece]).all()
    assert backtest["strategy_branch"] == "OUTSIDER_ONLY"



def test_empty_lag_frame_includes_legacy_velocity_column():
    empty = pd.DataFrame(columns=[
        "market_id", "recorded_at", "mid_price", "spread",
        "volume_5min", "price_velocity",
    ])

    result = add_lag_features(empty)

    assert "price_velocity_lag1" in result.columns


def test_velocity_interaction_fills_nan_and_supports_missing_column():
    common = {"mid_price": [0.8], "spread": [0.01], "time_left_min": [3.0]}
    with_nan = add_derived_features(pd.DataFrame({**common, "price_velocity": [np.nan]}))
    without_column = add_derived_features(pd.DataFrame(common))

    assert with_nan.loc[0, "velocity_x_phase"] == 0.0
    assert without_column.loc[0, "velocity_x_phase"] == 0.0


def test_velocity_lag_imputation_uses_only_prior_rows_in_each_market():
    frame = pd.DataFrame({
        "market_id": ["a", "a", "a", "b", "b"],
        "recorded_at": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
        "mid_price": [0.5] * 5,
        "spread": [0.01] * 5,
        "volume_5min": [1.0] * 5,
        "price_velocity": [0.1, np.nan, 0.3, 9.0, 8.0],
    })

    result = add_lag_features(frame)

    assert result["price_velocity_lag1"].tolist() == pytest.approx(
        [0.0, 0.1, 0.1, 0.0, 9.0]
    )
