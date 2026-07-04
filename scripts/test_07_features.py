import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

np.random.seed(42)
n = 150
price, t = 40000.0, datetime(2025, 1, 1, tzinfo=timezone.utc)
rows = []
for _ in range(n):
    o = price
    c = price * (1 + np.random.normal(0, 0.002))
    h = max(o, c) * (1 + abs(np.random.normal(0, 0.001)))
    l = min(o, c) * (1 - abs(np.random.normal(0, 0.001)))
    vol = abs(np.random.normal(100, 30))
    rows.append({
        "open_time": t, "open": o, "high": h, "low": l,
        "close": c, "volume": vol, "taker_buy_volume": vol * np.random.uniform(0.3, 0.7),
    })
    price, t = c, t + timedelta(minutes=15)

df = pd.DataFrame(rows)
result = build_crypto_features(df)

assert result.valid,                        "valid=False при 150 свечах"
assert result.features.shape == (1, len(CRYPTO_FEATURE_COLUMNS)), "Неверный shape"
assert not np.any(np.isnan(result.features)), "NaN в фичах!"
assert not np.any(np.isinf(result.features)), "Inf в фичах!"

for name, val in zip(CRYPTO_FEATURE_COLUMNS, result.features[0]):
    if "dist_to_high" in name: assert val <= 0.01, f"{name}={val:.4f}"
    if "dist_to_low"  in name: assert val >= -0.01, f"{name}={val:.4f}"
    if name == "hour_utc":     assert 0 <= val <= 23
    if name == "dow":          assert 0 <= val <= 6
    if name == "tbv_ratio":    assert 0.0 <= val <= 1.0, f"tbv_ratio={val:.4f}"

print(f"✅ FeatureBuilder OK — {len(CRYPTO_FEATURE_COLUMNS)} фичей")
