"""
tests/crypto/test_backtester_polymarket.py

Тесты бэктестера Polymarket:
  - Отсутствие 'target' в pnl_mode="polymarket" выбывает ValueError.
  - Одинаковое определение режимов low/mid/high через VolatilityRegimePolicy.
"""
import pytest
import pandas as pd
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.volatility import VolatilityRegimePolicy


def test_polymarket_backtest_without_target_raises_value_error():
    df_no_target = pd.DataFrame([
        {"open": 100.0, "close": 101.0, "vol_trend": 0.5}
    ])
    with pytest.raises(ValueError, match="Polymarket backtest requires canonical final_outcome target"):
        run_backtest(df_no_target, symbol="BTCUSDT", pnl_mode="polymarket")


def test_backtest_and_predictor_same_volatility_classification():
    policy = VolatilityRegimePolicy(low_boundary=0.8, high_boundary=1.2)
    assert policy.classify(0.5) == "low_vol"
    assert policy.classify(1.0) == "mid_vol"
    assert policy.classify(1.5) == "high_vol"


def test_sequence_backtest_requires_closed_candles():
    frame = pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "ret_1": [0.01, -0.01] * 4,
        "vol_trend": [1.0] * 8,
        "target": [0, 1] * 4,
    })
    with pytest.raises(ValueError, match="requires closed_candles"):
        run_backtest(frame, symbol="BTCUSDT", feature_set="B")
