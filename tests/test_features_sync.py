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
    critical = ["ret_1", "rsi_14", "vol_6"]
    for col in critical:
        if col in out.columns:
            assert out[col].notna().any(), f"{col} — все значения NaN"


def test_batch_and_inference_features_match_exactly():
    """batch.iloc[-1] и build_crypto_features совпадают по всем общим полям: rtol=1e-8, atol=1e-10."""
    import numpy as np
    from polyflip.crypto.feature_builder import (
        build_features,
        build_crypto_features,
        CRYPTO_FEATURE_COLUMNS,
    )
    from tests.fixtures.model_audit.audit_data import generate_variable_volatility_candles

    candles_df = generate_variable_volatility_candles(120)
    batch_df = build_features(candles_df, ddof=1)
    inference_res = build_crypto_features(candles_df, ddof=1)

    assert inference_res.valid is True
    last_batch_row = batch_df.iloc[-1]
    inference_vec = inference_res.features[0]

    # Compare each common column
    common_cols = [c for c in CRYPTO_FEATURE_COLUMNS if c in batch_df.columns]
    for col in common_cols:
        idx = CRYPTO_FEATURE_COLUMNS.index(col)
        batch_val = float(last_batch_row[col])
        inf_val = float(inference_vec[idx])
        assert np.isclose(batch_val, inf_val, rtol=1e-8, atol=1e-10), (
            f"Feature {col} mismatch: batch={batch_val:.10f}, inference={inf_val:.10f}, "
            f"diff={abs(batch_val - inf_val):.2e}"
        )


def test_batch_inference_match_edge_cases():
    """Проверка edge cases: постоянная цена, нулевой объем, минимальная история."""
    import numpy as np, pandas as pd
    from datetime import datetime, timedelta, timezone
    from polyflip.crypto.feature_builder import (
        build_features,
        build_crypto_features,
        CRYPTO_FEATURE_COLUMNS,
    )

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # 1. Constant price
    n = 100
    const_df = pd.DataFrame({
        "open_time": [t0 + timedelta(minutes=15 * i) for i in range(n)],
        "open": [50000.0] * n,
        "high": [50000.0] * n,
        "low": [50000.0] * n,
        "close": [50000.0] * n,
        "volume": [10.0] * n,
        "taker_buy_volume": [5.0] * n,
    })
    b_const = build_features(const_df, ddof=1)
    inf_const = build_crypto_features(const_df, min_candles=100, ddof=1)
    assert inf_const.valid is True
    for col in ["vol_6", "vol_24", "vol_trend", "ret_1", "bb_width"]:
        idx = CRYPTO_FEATURE_COLUMNS.index(col)
        assert np.isclose(b_const.iloc[-1][col], inf_const.features[0][idx], rtol=1e-8, atol=1e-10)

    # 2. Zero volume
    zero_vol_df = const_df.copy()
    zero_vol_df["volume"] = 0.0
    zero_vol_df["taker_buy_volume"] = 0.0
    b_zero = build_features(zero_vol_df, ddof=1)
    inf_zero = build_crypto_features(zero_vol_df, min_candles=100, ddof=1)
    for col in ["vol_z_1", "taker_buy_ratio", "cvd_1", "cvd_6"]:
        idx = CRYPTO_FEATURE_COLUMNS.index(col)
        assert np.isclose(b_zero.iloc[-1][col], inf_zero.features[0][idx], rtol=1e-8, atol=1e-10)


def test_btc_routing_discrepancy_eliminated():
    """
    На BTC точках разница вычисления волатильности из-за двух реализаций
    должна стать 0 вместо прежних расхождений из-за ddof=0 vs ddof=1.
    """
    import numpy as np
    from polyflip.crypto.feature_builder import (
        build_features,
        build_crypto_features,
        CRYPTO_FEATURE_COLUMNS,
    )
    from polyflip.crypto.volatility import VolatilityRegimePolicy
    from tests.fixtures.model_audit.audit_data import generate_btc_800_integration_points

    btc_df = generate_btc_800_integration_points()
    vol_policy = VolatilityRegimePolicy(low_boundary=0.8, high_boundary=1.2)

    vol_trend_idx = CRYPTO_FEATURE_COLUMNS.index("vol_trend")

    # Check 100 rolling points
    mismatches = 0
    tested = 0
    for end_idx in range(100, 200):
        sub_df = btc_df.iloc[:end_idx]
        b = build_features(sub_df, ddof=1)
        inf = build_crypto_features(sub_df, min_candles=100, ddof=1)
        b_trend = float(b.iloc[-1]["vol_trend"])
        inf_trend = float(inf.features[0][vol_trend_idx])
        assert np.isclose(b_trend, inf_trend, rtol=1e-8, atol=1e-10)

        reg_b = vol_policy.classify(b_trend)
        reg_inf = vol_policy.classify(inf_trend)
        if reg_b != reg_inf:
            mismatches += 1
        tested += 1

    assert mismatches == 0, f"Routing mismatches found: {mismatches}/{tested}"


def test_ddof_discrepancy_reproduced_with_legacy():
    """Подтверждаем исходный баг: ddof=0 (legacy) vs ddof=1 дает расхождение ~9.5% на 6-свечном окне."""
    import numpy as np
    from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS
    from tests.fixtures.model_audit.audit_data import generate_variable_volatility_candles

    candles_df = generate_variable_volatility_candles(120)
    res_ddof1 = build_crypto_features(candles_df, ddof=1)
    res_ddof0 = build_crypto_features(candles_df, ddof=0)

    vol_6_idx = CRYPTO_FEATURE_COLUMNS.index("vol_6")
    v6_1 = res_ddof1.features[0][vol_6_idx]
    v6_0 = res_ddof0.features[0][vol_6_idx]

    # Ratio should be sqrt(6 / 5) = 1.095445...
    assert v6_1 > v6_0
    ratio = v6_1 / v6_0
    assert np.isclose(ratio, np.sqrt(6.0 / 5.0), rtol=1e-4)


