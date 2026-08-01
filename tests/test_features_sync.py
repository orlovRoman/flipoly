def test_crypto_features_subset_of_feature_columns():
    """CRYPTO_FEATURES должен быть строгим подмножеством CRYPTO_FEATURE_COLUMNS."""
    from polyflip.crypto.trainer import CRYPTO_FEATURES
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
    unknown = set(CRYPTO_FEATURES) - set(CRYPTO_FEATURE_COLUMNS)
    assert not unknown, f"Неизвестные фичи: {unknown}"


def test_feature_builder_produces_all_features():
    """build_features() должен содержать все CRYPTO_FEATURES как колонки."""
    import numpy as np, pandas as pd
    from polyflip.crypto.trainer import CRYPTO_FEATURES
    from polyflip.crypto.feature_builder import build_features

    # Минимальный синтетический датасет
    n = 120
    np.random.seed(0)
    base = 50000 + np.cumsum(np.random.randn(n) * 100)
    from datetime import datetime, timedelta, timezone
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
    out = build_features(candles_df)
    missing = [f for f in CRYPTO_FEATURES if f not in out.columns]
    assert not missing, f"Отсутствуют в build_features: {missing}"

    # Проверяем что ключевые фичи не all-NaN
    critical = ["ret_1", "vol_ratio", "rsi_14", "dist_to_high_96", "vol_z_6"]
    for col in critical:
        if col in out.columns:
            assert out[col].notna().any(), f"{col} — все значения NaN"
