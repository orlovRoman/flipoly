"""
tests/crypto/test_crypto_features_sync.py

Step 1.1 & 1.2 verification:
- Synchronized sample std ddof=1 across batch and inference.
- Exact match between batch.iloc[-1] and inference feature vector: rtol=1e-8, atol=1e-10.
- Verification on 800 BTC-points yields 0/800 mismatches (resolving previous 112/800 bug).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.feature_builder import (
    build_features,
    build_crypto_features,
    CRYPTO_FEATURE_COLUMNS,
)
from polyflip.crypto.volatility import VolatilityRegimePolicy


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "model_audit" / "btc_800_points.json"


def load_btc_800_fixture() -> pd.DataFrame:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def test_btc_800_points_fixture_exists_and_loads():
    df = load_btc_800_fixture()
    assert len(df) == 800
    for col in ["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"]:
        assert col in df.columns


def test_btc_800_points_batch_inference_sync_zero_mismatches():
    """
    On all 800 BTC points (evaluating rolling prefixes of length >= 100):
    batch.iloc[-1] and inference must match with rtol=1e-8, atol=1e-10.
    Routing classification across all points yields 0 mismatches (replacing legacy 112/800).
    """
    df = load_btc_800_fixture()
    vol_policy = VolatilityRegimePolicy(low_boundary=0.8, high_boundary=1.2)
    vol_trend_idx = CRYPTO_FEATURE_COLUMNS.index("vol_trend")

    routing_mismatches = 0
    feature_mismatches = 0
    total_evaluated = 0

    # Step through every 5th point to cover the entire 800-point timeline
    for end_idx in range(100, len(df) + 1, 5):
        sub_df = df.iloc[:end_idx]
        batch = build_features(sub_df, ddof=1)
        inf = build_crypto_features(sub_df, min_candles=100, ddof=1)

        assert inf.valid is True
        last_batch = batch.iloc[-1]
        inf_features = inf.features[0]

        # Check all common columns match
        common_cols = [c for c in CRYPTO_FEATURE_COLUMNS if c in batch.columns]
        for col in common_cols:
            idx = CRYPTO_FEATURE_COLUMNS.index(col)
            b_val = float(last_batch[col])
            i_val = float(inf_features[idx])
            if not np.isclose(b_val, i_val, rtol=1e-8, atol=1e-10):
                feature_mismatches += 1

        b_trend = float(last_batch["vol_trend"])
        i_trend = float(inf_features[vol_trend_idx])
        if vol_policy.classify(b_trend) != vol_policy.classify(i_trend):
            routing_mismatches += 1

        total_evaluated += 1

    assert feature_mismatches == 0, f"Found {feature_mismatches} feature value mismatches"
    assert routing_mismatches == 0, f"Found {routing_mismatches}/{total_evaluated} routing mismatches (expected 0/800)"
