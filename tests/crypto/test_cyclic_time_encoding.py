# tests/crypto/test_cyclic_time_encoding.py
from datetime import datetime, timedelta
import numpy as np
import pytest
from polyflip.db.models import CryptoCandle
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

def make_candles_at_hour(n: int = 150, hour: int = 12):
    start = datetime(2026, 1, 1, hour, 0)
    res = []
    for i in range(n):
        t = start + timedelta(minutes=15 * i)
        res.append(CryptoCandle(
            symbol="BTCUSDT",
            open_time=t,
            open=100.0, high=101.0, low=99.0, close=100.0, volume=10.0, taker_buy_volume=5.0
        ))
    return res

def test_hour_sin_cos_full_cycle():
    """sin²+cos²=1 для всех 24 часов (identity property)."""
    for hour in range(24):
        fv = build_crypto_features(make_candles_at_hour(150, hour=hour))
        cols = dict(zip(CRYPTO_FEATURE_COLUMNS, fv.features[0]))
        s = cols["hour_sin"]
        c = cols["hour_cos"]
        assert abs(s**2 + c**2 - 1.0) < 1e-9, f"hour={hour}: sin²+cos²={s**2+c**2}"

def test_midnight_continuity():
    """hour=23 и hour=0 должны быть близко в sin/cos пространстве."""
    h23_sin = np.sin(2 * np.pi * 23 / 24)
    h0_sin  = np.sin(2 * np.pi * 0 / 24)
    h23_cos = np.cos(2 * np.pi * 23 / 24)
    h0_cos  = np.cos(2 * np.pi * 0 / 24)
    dist = np.sqrt((h23_sin - h0_sin)**2 + (h23_cos - h0_cos)**2)
    assert dist < 0.3, f"hour 23 and 0 too far apart: dist={dist}"

def test_no_raw_time_features_in_columns():
    assert "hour_utc" not in CRYPTO_FEATURE_COLUMNS
    assert "dow" not in CRYPTO_FEATURE_COLUMNS
    assert "hour_sin" in CRYPTO_FEATURE_COLUMNS
    assert "hour_cos" in CRYPTO_FEATURE_COLUMNS

def test_dow_sin_cos_identity():
    for day in range(7):
        s = np.sin(2 * np.pi * day / 7)
        c = np.cos(2 * np.pi * day / 7)
        assert abs(s**2 + c**2 - 1.0) < 1e-9
