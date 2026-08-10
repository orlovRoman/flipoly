import numpy as np
import pandas as pd
import pytest

from polyflip.models.trainer import _compute_backtest_pnl, add_derived_features
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
