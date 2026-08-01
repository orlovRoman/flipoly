def test_build_crypto_features_requires_100_candles():
    """При < 100 свечах возвращает valid=False."""
    import numpy as np, pandas as pd
    from datetime import datetime, timedelta, timezone
    from polyflip.crypto.feature_builder import build_crypto_features

    def make_candles(n):
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return pd.DataFrame({
            "open_time":        [t0 + timedelta(minutes=15*i) for i in range(n)],
            "open":  np.ones(n) * 50000,
            "high":  np.ones(n) * 50100,
            "low":   np.ones(n) * 49900,
            "close": np.ones(n) * 50000,
            "volume": np.ones(n) * 10,
            "taker_buy_volume": np.ones(n) * 5,
        })

    assert build_crypto_features(make_candles(99)).valid is False
    assert build_crypto_features(make_candles(100)).valid is True
