# tests/crypto/test_feature_builder_horizons.py
from datetime import datetime, timedelta
import pandas as pd
from polyflip.db.models import CryptoCandle

def make_candles(n: int = 150):
    start = datetime(2026, 1, 1, 0, 0)
    res = []
    for i in range(n):
        t = start + timedelta(minutes=15 * i)
        res.append(CryptoCandle(
            symbol="BTCUSDT",
            open_time=t,
            open=100.0 + i * 0.1,
            high=100.5 + i * 0.1,
            low=99.5 + i * 0.1,
            close=100.2 + i * 0.1,
            volume=10.0,
            taker_buy_volume=6.0,
        ))
    return res

def test_no_long_horizon_features_in_columns():
    """Гарантируем что длинные горизонты не вернулись в COLUMNS."""
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
    banned = {"ret_48", "ret_24", "ret_12", "vol_48", "vol_ratio",
              "dist_to_high_96", "dist_to_low_96"}
    found = banned & set(CRYPTO_FEATURE_COLUMNS)
    assert not found, f"Long-horizon features found: {found}"

def test_feature_vector_shape_matches_columns():
    """Shape вектора == len(CRYPTO_FEATURE_COLUMNS) после удаления фич."""
    from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS
    fv = build_crypto_features(make_candles(150))
    assert fv.features.shape == (1, len(CRYPTO_FEATURE_COLUMNS))

def test_build_features_no_long_horizon_columns():
    """build_features() не должен содержать удалённые колонки."""
    from polyflip.crypto.feature_builder import build_features
    df = build_features(make_candles(150))
    for col in ["ret_48", "vol_48", "dist_to_high_96"]:
        assert col not in df.columns, f"Column {col} should be removed"

def test_feature_count_regression():
    """Счётчик фич — явная фиксация для ловли случайных изменений."""
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
    assert len(CRYPTO_FEATURE_COLUMNS) == 25, \
        f"Expected 25 features, got {len(CRYPTO_FEATURE_COLUMNS)}"
