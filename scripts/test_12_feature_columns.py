# scripts/test_12_feature_columns.py
from polyflip.crypto.feature_builder import (
    build_crypto_features, build_features, CRYPTO_FEATURE_COLUMNS
)
from polyflip.crypto.trainer import CRYPTO_FEATURES
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

def _make_candles(n=200):
    """Генерирует синтетические свечи."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 50_000.0
    for i in range(n):
        change = 1 + np.random.uniform(-0.005, 0.005)
        price *= change
        rows.append({
            "open_time": base + timedelta(minutes=15 * i),
            "open": price * 0.999,
            "high": price * 1.002,
            "low":  price * 0.997,
            "close": price,
            "volume": np.random.uniform(100, 1000),
            "taker_buy_volume": np.random.uniform(40, 600),
        })
    return pd.DataFrame(rows)

df = _make_candles(200)
out = build_features(df)

# Тест 1: все тренировочные фичи присутствуют в build_features()
missing_train = [f for f in CRYPTO_FEATURES if f not in out.columns]
assert not missing_train, f"Отсутствуют в build_features: {missing_train}"

# Тест 2: онлайн-вектор имеет правильный shape
vec = build_crypto_features(df)
assert vec.valid, "build_crypto_features должен вернуть valid=True"
assert vec.features.shape == (1, len(CRYPTO_FEATURE_COLUMNS)), \
    f"Shape {vec.features.shape} != (1, {len(CRYPTO_FEATURE_COLUMNS)})"

# Тест 3: нет NaN/Inf
assert not np.any(np.isnan(vec.features)), "NaN в онлайн-векторе"
assert not np.any(np.isinf(vec.features)), "Inf в онлайн-векторе"

# Тест 4: ret_48 есть в build_features()
assert "ret_48" in out.columns, "ret_48 отсутствует в build_features()"

# Тест 5: соответствие имен в CRYPTO_FEATURES и CRYPTO_FEATURE_COLUMNS
# trainer использует подмножество фичей из feature_builder.
# Убедимся, что все CRYPTO_FEATURES есть в CRYPTO_FEATURE_COLUMNS
missing_in_columns = [f for f in CRYPTO_FEATURES if f not in CRYPTO_FEATURE_COLUMNS]
assert not missing_in_columns, f"Фичи из trainer отсутствуют в CRYPTO_FEATURE_COLUMNS: {missing_in_columns}"

print("✅ Тест 12: feature columns OK")
