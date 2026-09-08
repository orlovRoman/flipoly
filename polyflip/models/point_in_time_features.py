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


def compute_point_in_time_features(
    df: pd.DataFrame,
    decision_at: datetime | None = None,
    max_reference_age_60s: float = TOLERANCE_60S_MAX,
    max_reference_age_180s: float = TOLERANCE_180S_MAX,
) -> pd.DataFrame:
    """
    Computes time-based dynamic features for each market.
    Guarantees prefix invariance and point-in-time causality.
    """
    if df.empty:
        res = df.copy()
        for col in [
            "pm_change_60s", "pm_change_180s", "legacy_last_poll_delta",
            "price_distance_from_max", "has_60s_ref", "has_180s_ref", "history_age_seconds"
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

    # If decision_at is specified, strictly exclude any rows beyond decision_at
    if decision_at is not None:
        dec_utc = decision_at if getattr(decision_at, "tzinfo", None) is not None else decision_at.replace(tzinfo=timezone.utc)
        out["_is_causal"] = out["_rec_dt"] <= dec_utc
    else:
        out["_is_causal"] = True

    market_col = "market_id" if "market_id" in out.columns else "_market_dummy"
    if market_col not in out.columns:
        out[market_col] = "m_default"

    # Sort deterministically
    out = out.sort_values([market_col, "_rec_dt"]).reset_index(drop=True)

    pm_change_60s = []
    pm_change_180s = []
    has_60s_ref = []
    has_180s_ref = []
    legacy_last_poll_delta = []
    price_dist_from_max = []
    history_age_sec = []

    for group_key, grp in out.groupby(market_col, sort=False):
        times = grp["_rec_dt"].values
        prices = grp["mid_price"].values
        is_causal = grp["_is_causal"].values

        exp_max = 0.0
        for i in range(len(grp)):
            t_curr = times[i]
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
            price_dist_from_max.append(max(0.0, exp_max - p_curr))

            # History age
            earliest_t = times[0]
            age_s = float((t_curr - earliest_t) / np.timedelta64(1, "s"))
            history_age_sec.append(age_s)

            # Legacy poll delta
            if i > 0:
                legacy_last_poll_delta.append(p_curr - float(prices[i - 1]))
            else:
                legacy_last_poll_delta.append(0.0)

            # 60s as-of reference search: target ~60s ago
            # Acceptable age delta: [TOLERANCE_60S_MIN, max_reference_age_60s]
            val_60 = np.nan
            has_60 = 0.0
            best_diff_60 = float("inf")
            for j in range(i - 1, -1, -1):
                delta_s = float((t_curr - times[j]) / np.timedelta64(1, "s"))
                if delta_s < TOLERANCE_60S_MIN:
                    continue
                if delta_s > max_reference_age_60s:
                    break
                diff = abs(delta_s - 60.0)
                if diff < best_diff_60:
                    best_diff_60 = diff
                    val_60 = p_curr - float(prices[j])
                    has_60 = 1.0

            pm_change_60s.append(val_60)
            has_60s_ref.append(has_60)

            # 180s as-of reference search: target ~180s ago
            # Acceptable age delta: [TOLERANCE_180S_MIN, max_reference_age_180s]
            val_180 = np.nan
            has_180 = 0.0
            best_diff_180 = float("inf")
            for j in range(i - 1, -1, -1):
                delta_s = float((t_curr - times[j]) / np.timedelta64(1, "s"))
                if delta_s < TOLERANCE_180S_MIN:
                    continue
                if delta_s > max_reference_age_180s:
                    break
                diff = abs(delta_s - 180.0)
                if diff < best_diff_180:
                    best_diff_180 = diff
                    val_180 = p_curr - float(prices[j])
                    has_180 = 1.0

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
