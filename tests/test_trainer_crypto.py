import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from polyflip.crypto.trainer import _fit_lgbm_and_serialize, _build_target, CRYPTO_FEATURES

def make_fake_df(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "ret_1":           np.random.randn(n) * 0.003,
        "ret_3":           np.random.randn(n) * 0.005,
        "ret_6":           np.random.randn(n) * 0.007,
        "ret_12":          np.random.randn(n) * 0.009,
        "ret_24":          np.random.randn(n) * 0.012,
        "vol_6":           np.abs(np.random.randn(n)) * 0.002 + 0.001,
        "vol_24":          np.abs(np.random.randn(n)) * 0.003 + 0.001,
        "vol_48":          np.abs(np.random.randn(n)) * 0.003 + 0.001,
        "vol_ratio":       np.random.uniform(0.5, 2.0, n),
        "rsi_14":          np.random.uniform(30, 70, n),
        "ema_ratio_9_21":  np.random.uniform(0.99, 1.01, n),
        "bb_width":        np.random.uniform(0.01, 0.05, n),
        "bb_position":     np.random.uniform(0, 1, n),
        "taker_buy_ratio": np.random.uniform(0.4, 0.6, n),
        "hour_utc":        np.random.randint(0, 24, n).astype(float),
        "consec_up":       np.random.randint(0, 5, n).astype(float),
        "consec_down":     np.random.randint(0, 5, n).astype(float),
    })
    return df

def test_vol_regime_split():
    """low_vol и high_vol датасеты не пересекаются."""
    df = make_fake_df(500)
    median = df["vol_ratio"].median()
    low  = df[df["vol_ratio"] <= median]
    high = df[df["vol_ratio"] > median]
    assert len(low) + len(high) == len(df)
    assert set(low.index) & set(high.index) == set()
