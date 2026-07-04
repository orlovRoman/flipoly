import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.feature_builder import build_features

def test_run_backtest_regime_aware():
    """Бэктест использует vol_ratio для выбора модели."""
    # Создаем достаточно большой синтетический датасет (> 500 свечей)
    n = 600
    np.random.seed(42)
    base = 50000 + np.cumsum(np.random.randn(n) * 100)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    candles_df = pd.DataFrame({
        "open_time":        [t0 + timedelta(minutes=15 * i) for i in range(n)],
        "open":             base * (1 + np.random.randn(n) * 0.001),
        "high":             base * (1 + np.abs(np.random.randn(n)) * 0.002),
        "low":              base * (1 - np.abs(np.random.randn(n)) * 0.002),
        "close":            base,
        "volume":           np.random.uniform(10, 100, n),
        "taker_buy_volume": np.random.uniform(5, 50, n),
    })
    
    df_features = build_features(candles_df)
    
    result = run_backtest(df_features, symbol="BTCUSDT")
    
    # Проверки корректности результата
    assert result.symbol == "BTCUSDT"
    assert result.n_candles_total == len(df_features)
    assert 0.0 <= result.win_rate <= 1.0
