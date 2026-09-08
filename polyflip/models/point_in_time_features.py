"""
polyflip/models/point_in_time_features.py

Point-in-time dynamic features calculated over explicit temporal horizons (seconds)
rather than discrete snapshot row shifts.

Features:
- pm_change_60s: mid_price change as-of ~60s ago
- pm_change_180s: mid_price change as-of ~180s ago
- legacy_last_poll_delta: delta from immediately preceding poll (formerly price_velocity)
- price_distance_from_max: expanding max of mid_price from market open up to decision_at minus current mid_price
- history_age_seconds: age of earliest available snapshot for this market
- has_60s_ref, has_180s_ref: boolean/float flags indicating whether valid reference points exist
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Sequence

# Tolerance windows for as-of lookup (in seconds)
TOLERANCE_60S_MIN = 30.0
TOLERANCE_60S_MAX = 90.0

TOLERANCE_180S_MIN = 120.0
TOLERANCE_180S_MAX = 240.0

POINT_IN_TIME_FEATURE_NAMES: tuple[str, ...] = (
    "pm_change_60s",
    "pm_change_180s",
    "has_60s_ref",
    "has_180s_ref",
    "legacy_last_poll_delta",
    "price_distance_from_max",
    "history_age_seconds",
)


def apply_market_feature_pipeline(
    df: pd.DataFrame,
    decision_at: datetime | None = None,
    global_max: float | None = None,
) -> pd.DataFrame:
    """
    Unified feature pipeline applied identically in both training (ModelTrainer.train_model)
    and live inference (build_inference_dataframe).
    Chains:
      1. add_derived_features
      2. add_lag_features
      3. compute_point_in_time_features
    Guarantees complete train/serve parity and prefix invariance.
    """
    from polyflip.models.trainer import add_derived_features
    from polyflip.models.feature_lags import add_lag_features

    df = add_derived_features(df)
    df = add_lag_features(df)
    df = compute_point_in_time_features(df, decision_at=decision_at, global_max=global_max)
    return df


def compute_point_in_time_features(
    df: pd.DataFrame,
    decision_at: datetime | None = None,
    max_reference_age_60s: float = TOLERANCE_60S_MAX,
    max_reference_age_180s: float = TOLERANCE_180S_MAX,
    global_max: float | None = None,
) -> pd.DataFrame:
    """
    Computes time-based dynamic features for each market.
    Guarantees prefix invariance and point-in-time causality.
    """
    if df.empty:
        res = df.copy()
        for col in [
            "pm_change_60s", "pm_change_180s", "legacy_last_poll_delta",
            "price_distance_from_max", "has_60s_ref", "has_180s_ref",
            "history_age_seconds", "price_velocity",
        ]:
            res[col] = 0.0
        return res

    out = df.copy()
    orig_index = out.index
    out["_orig_row_pos"] = np.arange(len(out))

    # Ensure recorded_at is timezone-aware UTC datetime
    if "recorded_at" in out.columns:
        out["_rec_dt"] = pd.to_datetime(out["recorded_at"], utc=True)
    else:
        out["_rec_dt"] = pd.date_range("2026-01-01", periods=len(out), freq="1min", tz="UTC")

    # If some values are NaT, fill them causally
    if out["_rec_dt"].isna().any():
        if decision_at is not None:
            dec_dt = pd.to_datetime(decision_at, utc=True)
            if "time_left_min" in out.columns:
                dec_rows = out[out["_is_decision_row"]] if "_is_decision_row" in out.columns else pd.DataFrame()
                dec_time_left = float(dec_rows["time_left_min"].iloc[0]) if len(dec_rows) > 0 else 0.0
                deltas = (pd.to_numeric(out["time_left_min"], errors="coerce").fillna(0.0) - dec_time_left)
                imputed = [dec_dt - timedelta(minutes=float(d)) if float(d) > 0 else dec_dt for d in deltas]
                out["_rec_dt"] = out["_rec_dt"].fillna(pd.Series(imputed, index=out.index))
        out["_rec_dt"] = out["_rec_dt"].bfill().ffill()
        if out["_rec_dt"].isna().any():
            out["_rec_dt"] = out["_rec_dt"].fillna(pd.Timestamp.now(tz=timezone.utc))

    # If decision_at is specified, strictly exclude any rows beyond decision_at
    if decision_at is not None:
        dec_utc = pd.to_datetime(decision_at, utc=True)
        out["_is_causal"] = out["_rec_dt"] <= dec_utc
    else:
        out["_is_causal"] = True

    market_col = "market_id" if "market_id" in out.columns else "_market_dummy"
    if market_col not in out.columns:
        out[market_col] = "m_default"
    else:
        out[market_col] = out[market_col].replace("", "m_default").fillna("m_default")

    # Sort deterministically: market, recorded_at, decision row last on collision, original order tiebreaker
    sort_cols = [market_col, "_rec_dt"]
    if "_is_decision_row" in out.columns:
        sort_cols.append("_is_decision_row")
    sort_cols.append("_orig_row_pos")
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    pm_change_60s = []
    pm_change_180s = []
    has_60s_ref = []
    has_180s_ref = []
    legacy_last_poll_delta = []
    price_dist_from_max = []
    history_age_sec = []

    def _find_as_of_reference(
        times_sec: np.ndarray,
        prices: np.ndarray,
        i: int,
        t_curr_sec: float,
        p_curr: float,
        target_sec: float,
        tolerance_min: float,
        max_reference_age: float,
    ) -> tuple[float, float]:
        if i <= 0:
            return np.nan, 0.0
        t_min = t_curr_sec - max_reference_age
        t_max = t_curr_sec - tolerance_min
        if t_max < 0:
            return np.nan, 0.0

        left = int(np.searchsorted(times_sec[:i], t_min, side="left"))
        right = int(np.searchsorted(times_sec[:i], t_max, side="right"))
        if left >= right:
            return np.nan, 0.0

        target_t = t_curr_sec - target_sec
        idx = int(np.searchsorted(times_sec[left:right], target_t, side="right"))
        best_j = -1
        best_diff = float("inf")
        for cand in (left + idx, left + idx - 1):
            if left <= cand < right:
                diff = abs((t_curr_sec - times_sec[cand]) - target_sec)
                if diff < best_diff:
                    best_diff = diff
                    best_j = cand

        if best_j >= 0:
            return p_curr - float(prices[best_j]), 1.0
        return np.nan, 0.0

    for group_key, grp in out.groupby(market_col, sort=False):
        times = grp["_rec_dt"].values
        prices = grp["mid_price"].values
        is_causal = grp["_is_causal"].values
        times_sec = (times - times[0]) / np.timedelta64(1, "s") if len(times) > 0 else np.array([])

        exp_max = 0.0
        for i in range(len(grp)):
            t_curr = times[i]
            t_curr_sec = float(times_sec[i])
            p_curr = float(prices[i])

            if not is_causal[i]:
                # Future row relative to decision_at
                pm_change_60s.append(np.nan)
                pm_change_180s.append(np.nan)
                has_60s_ref.append(0.0)
                has_180s_ref.append(0.0)
                legacy_last_poll_delta.append(0.0)
                price_dist_from_max.append(0.0)
                history_age_sec.append(0.0)
                continue

            # Update expanding max causal to this row
            exp_max = max(exp_max, p_curr)
            eff_max = max(exp_max, float(global_max)) if (global_max is not None and np.isfinite(global_max) and global_max > 0.0) else exp_max
            price_dist_from_max.append(max(0.0, eff_max - p_curr))

            # History age
            earliest_t = times[0]
            age_s = float((t_curr - earliest_t) / np.timedelta64(1, "s"))
            history_age_sec.append(age_s)

            # Legacy poll delta
            if i > 0:
                legacy_last_poll_delta.append(p_curr - float(prices[i - 1]))
            else:
                initial_vel = (
                    float(grp["price_velocity"].iloc[0])
                    if ("price_velocity" in grp.columns and pd.notna(grp["price_velocity"].iloc[0]))
                    else 0.0
                )
                legacy_last_poll_delta.append(initial_vel)

            # 60s as-of reference search: target ~60s ago via binary search
            val_60, has_60 = _find_as_of_reference(
                times_sec=times_sec,
                prices=prices,
                i=i,
                t_curr_sec=t_curr_sec,
                p_curr=p_curr,
                target_sec=60.0,
                tolerance_min=TOLERANCE_60S_MIN,
                max_reference_age=max_reference_age_60s,
            )
            pm_change_60s.append(val_60)
            has_60s_ref.append(has_60)

            # 180s as-of reference search: target ~180s ago via binary search
            val_180, has_180 = _find_as_of_reference(
                times_sec=times_sec,
                prices=prices,
                i=i,
                t_curr_sec=t_curr_sec,
                p_curr=p_curr,
                target_sec=180.0,
                tolerance_min=TOLERANCE_180S_MIN,
                max_reference_age=max_reference_age_180s,
            )
            pm_change_180s.append(val_180)
            has_180s_ref.append(has_180)

    out["pm_change_60s"] = pm_change_60s
    out["pm_change_180s"] = pm_change_180s
    out["has_60s_ref"] = has_60s_ref
    out["has_180s_ref"] = has_180s_ref
    out["legacy_last_poll_delta"] = legacy_last_poll_delta
    out["price_distance_from_max"] = price_dist_from_max
    out["history_age_seconds"] = history_age_sec

    # Keep compatibility with price_velocity name
    out["price_velocity"] = out["legacy_last_poll_delta"]

    out = out.sort_values("_orig_row_pos").reset_index(drop=True)
    out = out.drop(columns=["_orig_row_pos", "_rec_dt", "_market_dummy", "_is_causal"], errors="ignore")
    out.index = orig_index
    return out
