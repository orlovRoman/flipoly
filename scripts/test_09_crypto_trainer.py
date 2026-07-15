import numpy as np
import pandas as pd
from polyflip.crypto.trainer import _fit_lgbm_and_serialize, _build_target, CRYPTO_FEATURES


def make_fake_df(n=1500):
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


df = make_fake_df()
df_filtered = _build_target(df)
assert len(df_filtered) > 100, f"После построения таргетов слишком мало строк: {len(df_filtered)}"

X = df_filtered[[f for f in CRYPTO_FEATURES if f in df_filtered.columns]]
y = df_filtered["target"]
model_bytes, auc, baseline, thr, ece, fi = _fit_lgbm_and_serialize(X, y, n_splits=3)

assert len(model_bytes) > 1000,     f"pickle пустой: {len(model_bytes)}"
assert 0.4 < auc < 1.0,             f"Странный AUC: {auc}"
assert 0.0 < thr < 1.0,             f"Странный порог: {thr}"
assert ece < 0.4,                   f"ECE слишком большой: {ece}"
assert isinstance(fi, dict) and len(fi) > 0, "Feature importance должна быть непустым словарем"

print(f"CryptoTrainer OK: AUC={auc:.3f}, thr={thr:.3f}, ECE={ece:.4f}")
print(f"  baseline={baseline:.3f}, model_bytes={len(model_bytes)}, features_in_fi={len(fi)}")
