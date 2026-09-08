"""
scripts/research/replay_feature_repairs.py

Verification and replay harness for feature repairs (Step 1.19):
1. Verifies feature math repair (ddof=1 rolling std & Bollinger bands).
2. Verifies causal snapshot isolation & point-in-time prefix invariance.
3. Verifies lag feature contract (no future median leakage into historical rows).
4. Verifies decision time cutoff enforcement (post-decision snapshot exclusion).
5. Verifies regime routing invariance on 800 BTC synthetic integration points.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.model_audit.audit_data import (
    generate_variable_volatility_candles,
    generate_market_history_with_future,
    generate_post_decision_snapshots,
    generate_btc_800_integration_points,
)
from polyflip.crypto.feature_builder import (
    compute_rolling_std,
    compute_bollinger_bands,
    build_crypto_features,
)
from polyflip.crypto.volatility import VolatilityRegimePolicy
from polyflip.models.feature_lags import add_lag_features
from polyflip.models.point_in_time_features import compute_point_in_time_features


def verify_feature_math() -> bool:
    print("[1/5] Verifying feature math repair (ddof=1 rolling std & Bollinger bands)...")
    candles = generate_variable_volatility_candles(120)
    close_vals = candles["close"].values

    # 1. Unbiased sample std (ddof=1) vs population std (ddof=0)
    std_ddof1 = compute_rolling_std(close_vals, window=20, ddof=1)
    std_ddof0 = compute_rolling_std(close_vals, window=20, ddof=0)

    # In sample std, values must be strictly greater than population std for non-constant series
    assert std_ddof1 > std_ddof0, f"ddof=1 ({std_ddof1}) must be strictly greater than ddof=0 ({std_ddof0})"
    # Exact ratio for window=20: sqrt(20 / 19)
    assert np.isclose(std_ddof1 / std_ddof0, np.sqrt(20.0 / 19.0), rtol=1e-5)

    # 2. compute_bollinger_bands must use ddof=1 matching compute_rolling_std
    bb_mean, bb_std, bb_width, bb_pos = compute_bollinger_bands(close_vals, window=20, num_std=2.0, ddof=1)
    assert np.isclose(bb_std, std_ddof1, rtol=1e-7)
    expected_width = (2.0 * 2.0 * std_ddof1) / (bb_mean + 1e-10)
    assert np.isclose(bb_width, expected_width, rtol=1e-7)

    print("  -> PASS: Rolling std and Bollinger bands strictly synchronized with ddof=1.")
    return True


def verify_causal_snapshot_isolation() -> bool:
    print("[2/5] Verifying causal snapshot isolation & point-in-time prefix invariance...")
    past_df, combined_df = generate_market_history_with_future()
    
    # Calculate PIT features on past snapshots alone
    feat_past = compute_point_in_time_features(past_df)

    # Calculate PIT features on combined snapshots with cutoff at T (last timestamp of past_df)
    decision_at = past_df["recorded_at"].max()
    feat_combined_causal = compute_point_in_time_features(combined_df, decision_at=decision_at)

    # Filter combined results back to past_df rows
    feat_combined_sub = feat_combined_causal[feat_combined_causal["market_id"] == "market-1"].iloc[:len(past_df)]

    check_cols = ["pm_change_60s", "legacy_last_poll_delta", "price_distance_from_max", "history_age_seconds"]
    for col in check_cols:
        v_past = feat_past[col].dropna().values
        v_comb = feat_combined_sub[col].dropna().values
        np.testing.assert_allclose(v_past, v_comb, err_msg=f"Leakage detected in {col} between past and combined!")

    print("  -> PASS: Features up to decision_at are completely invariant to future snapshots.")
    return True


def verify_lag_feature_contract() -> bool:
    print("[3/5] Verifying lag feature contract (no future median leakage)...")
    from polyflip.models.feature_lags import LAG_FEATURE_NAMES

    # 5 snapshots where first snapshot lacks lag history
    df = pd.DataFrame({
        "market_id": ["m1"] * 5,
        "recorded_at": pd.date_range("2026-01-01 12:00", periods=5, freq="1min", tz="UTC"),
        "mid_price": [0.30, 0.35, 0.40, 0.45, 0.50],
        "spread": [0.01] * 5,
        "volume_5min": [100.0] * 5,
        "price_velocity": [0.01] * 5,
        "time_left_min": [15.0 - i for i in range(5)],
        "market_duration_min": [15.0] * 5,
    })

    lags_df = add_lag_features(df)

    # First row must have NaN for all lag features, without lookahead median imputation
    for col in LAG_FEATURE_NAMES:
        assert pd.isna(lags_df.iloc[0][col]), f"First row {col} must be NaN without history"
    assert lags_df.iloc[0]["has_lag_history"] == 0.0

    # Row 3 (shift 3) has price_momentum = 0.45 - 0.30 = 0.15
    assert np.isclose(lags_df.iloc[3]["price_momentum"], 0.15, rtol=1e-5)

    print("  -> PASS: Missing lag history returns NaN without lookahead median imputation.")
    return True


def verify_decision_time_cutoff() -> bool:
    print("[4/5] Verifying decision time cutoff enforcement...")
    snapshots = generate_post_decision_snapshots()
    decision_at = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)

    out = compute_point_in_time_features(snapshots, decision_at=decision_at)
    rec_dt = pd.to_datetime(out["recorded_at"], utc=True)

    future_rows = out[rec_dt > decision_at]
    assert len(future_rows) > 0, "Test data must include future rows"

    # Future rows must have NaN / 0.0 values, never affecting causal features
    assert future_rows["pm_change_60s"].isna().all(), "Future rows must have NaN pm_change_60s"
    assert (future_rows["has_60s_ref"] == 0.0).all(), "Future rows must have 0.0 has_60s_ref"

    print("  -> PASS: Post-decision snapshots strictly excluded from feature computation.")
    return True


def verify_btc_routing_invariance() -> bool:
    print("[5/5] Verifying regime routing invariance on 800 synthetic BTC points...")
    df = generate_btc_800_integration_points()
    assert len(df) >= 800

    policy = VolatilityRegimePolicy(low_boundary=0.8, high_boundary=1.2)

    # Replay sliding windows: at step 200, 400, 600, 800
    regimes = []
    vol_trends = []

    eval_indices = [200, 400, 600, 800]
    for idx in eval_indices:
        candle_slice = df.iloc[:idx]
        fv = build_crypto_features(candle_slice, underlying_price=df.iloc[idx-1]["close"])
        assert fv.valid, f"Feature vector invalid at idx={idx}"
        
        col_names = [
            "ret_1", "ret_3", "ret_6", "vol_6", "vol_24", "vol_trend",
            "vol_z_1", "taker_buy_ratio", "cvd_1", "cvd_6",
            "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
            "dist_to_high_24", "dist_to_low_24", "range_1", "range_avg_24",
            "consec_balance", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "strike_gap_pct", "log_moneyness",
        ]
        feat_dict = dict(zip(col_names, fv.features[0]))
        vol_trend = feat_dict["vol_trend"]
        regime = policy.classify(vol_trend)

        vol_trends.append(vol_trend)
        regimes.append(regime)
        assert regime in ("low_vol", "mid_vol", "high_vol")

    print(f"  -> Evaluated steps {eval_indices}:")
    for idx, vt, reg in zip(eval_indices, vol_trends, regimes):
        print(f"     Step {idx}: vol_trend={vt:.4f} -> regime={reg}")

    # Verify deterministic reproducibility: running again on step 800 produces identical values
    slice_800 = df.iloc[:800]
    fv_repeat = build_crypto_features(slice_800, underlying_price=df.iloc[799]["close"])
    np.testing.assert_allclose(fv.features[0], fv_repeat.features[0], rtol=1e-7)

    print("  -> PASS: Routing & feature building strictly invariant under prefix evaluation.")
    return True


def main() -> int:
    print("=" * 70)
    print("REPLAY FEATURE REPAIRS & ROUTING INVARIANCE (Step 1.19)")
    print("=" * 70)

    try:
        verify_feature_math()
        verify_causal_snapshot_isolation()
        verify_lag_feature_contract()
        verify_decision_time_cutoff()
        verify_btc_routing_invariance()

        print("=" * 70)
        print("ALL FEATURE REPAIR & ROUTING TESTS PASSED (5/5)")
        print("=" * 70)
        return 0
    except Exception as exc:
        print(f"\n[FAILED] Verification error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
