import numpy as np
import pandas as pd
import pytest
from polyflip.crypto.feature_builder import (
    build_features,
    build_crypto_features,
    CRYPTO_FEATURE_COLUMNS,
)

def _make_candles(n: int = 120) -> pd.DataFrame:
    """Синтетические свечи с детерминированной ценой."""
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.002, n))
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open":   prices * 0.999,
        "high":   prices * 1.002,
        "low":    prices * 0.997,
        "close":  prices,
        "volume": np.random.uniform(100, 1000, n),
        "taker_buy_volume": np.random.uniform(40, 600, n),
    })

def test_vol_trend_in_columns():
    """vol_trend должна присутствовать в CRYPTO_FEATURE_COLUMNS."""
    assert "vol_trend" in CRYPTO_FEATURE_COLUMNS

def test_build_features_has_vol_trend():
    """build_features() должна возвращать колонку vol_trend."""
    df = build_features(_make_candles())
    assert "vol_trend" in df.columns

def test_vol_trend_non_negative():
    """vol_trend = vol_6 / vol_24 — всегда >= 0."""
    df = build_features(_make_candles())
    assert (df["vol_trend"].dropna() >= 0).all()

def test_vol_trend_high_volatility():
    """Когда краткосрочная vol выше долгосрочной, vol_trend > 1."""
    candles = _make_candles(120)
    # Последние 10 свечей — очень волатильные
    candles.loc[110:, "close"] *= np.array([1, 0.98, 1.03, 0.97, 1.04, 0.96, 1.05, 0.95, 1.06, 0.94])
    df = build_features(candles)
    assert df["vol_trend"].iloc[-1] > 1.0, "Ожидается vol_trend > 1 при всплеске волатильности"

def test_inference_vector_shape_with_vol_trend():
    """build_crypto_features() должна возвращать вектор правильного размера."""
    candles = _make_candles(150)
    result = build_crypto_features(candles)
    assert result.valid
    assert result.features.shape == (1, len(CRYPTO_FEATURE_COLUMNS))

def test_no_nans_in_features():
    """После build_features() не должно быть NaN и Inf."""
    df = build_features(_make_candles())
    numeric_cols = [c for c in CRYPTO_FEATURE_COLUMNS if c in df.columns]
    assert not df[numeric_cols].isnull().any().any()
    assert not np.isinf(df[numeric_cols].values).any()
