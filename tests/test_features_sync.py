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
    critical = ["ret_1", "rsi_14", "vol_6", "funding_rate", "funding_rate_ma3"]
    for col in critical:
        if col in out.columns:
            assert out[col].notna().any(), f"{col} — все значения NaN"


def test_crypto_features_validator_funding_rates():
    """Проверяем валидацию экстремальных и нормальных ставок фандинга в CryptoFeaturesValidator."""
    from polyflip.crypto.predictor import CryptoFeaturesValidator
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS

    feature_dict = {f: 0.1 for f in CRYPTO_FEATURE_COLUMNS}
    feature_dict["funding_rate"] = 0.05
    feature_dict["funding_rate_ma3"] = -0.01

    validated = CryptoFeaturesValidator(**feature_dict)
    assert validated.funding_rate == 0.05
    assert validated.funding_rate_ma3 == -0.01


def test_build_crypto_features_includes_live_funding():
    """build_crypto_features() включает живые ставки фандинга в сформированный вектор."""
    from datetime import datetime, timedelta, timezone
    import numpy as np, pandas as pd
    from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

    n = 120
    np.random.seed(0)
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

    res = build_crypto_features(candles_df, funding_rate=0.0015, funding_rate_ma3=-0.0005)
    assert res.valid
    assert res.features.shape == (1, len(CRYPTO_FEATURE_COLUMNS))
    fr_idx = CRYPTO_FEATURE_COLUMNS.index("funding_rate")
    fr_ma3_idx = CRYPTO_FEATURE_COLUMNS.index("funding_rate_ma3")
    assert res.features[0][fr_idx] == 0.0015
    assert res.features[0][fr_ma3_idx] == -0.0005

