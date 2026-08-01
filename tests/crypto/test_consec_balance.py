# tests/crypto/test_consec_balance.py
from datetime import datetime, timedelta
import pytest
from polyflip.db.models import CryptoCandle
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

def make_candles_custom(dirs: list[bool]):
    start = datetime(2026, 1, 1, 12, 0)
    res = []
    for i, is_up in enumerate(dirs):
        t = start + timedelta(minutes=15 * i)
        close_val = 101.0 if is_up else 99.0
        res.append(CryptoCandle(
            symbol="BTCUSDT", open_time=t,
            open=100.0, high=102.0, low=98.0, close=close_val,
            volume=10.0, taker_buy_volume=5.0
        ))
    return res

def test_consec_balance_sign_on_up_series():
    # 110 candles total, last 5 are UP
    dirs = [False] * 105 + [True] * 5
    fv = build_crypto_features(make_candles_custom(dirs), min_candles=100)
    idx = CRYPTO_FEATURE_COLUMNS.index("consec_balance")
    assert fv.features[0][idx] > 0

def test_consec_balance_sign_on_down_series():
    # 110 candles total, last 5 are DOWN
    dirs = [True] * 105 + [False] * 5
    fv = build_crypto_features(make_candles_custom(dirs), min_candles=100)
    idx = CRYPTO_FEATURE_COLUMNS.index("consec_balance")
    assert fv.features[0][idx] < 0

def test_consec_balance_zero_on_alternating():
    dirs = ([True, False] * 60)
    fv = build_crypto_features(make_candles_custom(dirs), min_candles=100)
    idx = CRYPTO_FEATURE_COLUMNS.index("consec_balance")
    assert abs(fv.features[0][idx]) <= 1.0

def test_no_separate_consec_features():
    assert "consec_up"   not in CRYPTO_FEATURE_COLUMNS
    assert "consec_down" not in CRYPTO_FEATURE_COLUMNS
    assert "consec_balance" in CRYPTO_FEATURE_COLUMNS

def test_feature_count_final():
    assert len(CRYPTO_FEATURE_COLUMNS) == 23, f"Expected 23 features, got {len(CRYPTO_FEATURE_COLUMNS)}"
