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


def compute_normalized_strike_distance(
    underlying_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    sigma_1m: float | np.ndarray,
    tau_min: float | np.ndarray,
    candidate_side: str | Sequence[str],
    clip_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray | float, np.ndarray | bool]:
    """
    Computes normalized distance to strike (Item 2.2):
      z_outsider = s * log(S_t / K) / (sigma_1m * sqrt(tau_min))
    where s = +1 for candidate UP (or YES on UP strike), -1 for DOWN.

    Properties:
    - S_t == K => z = 0
    - UP vs DOWN flips sign: z_DOWN = -z_UP
    - Increasing tau or sigma decreases |z|
    - Safe handling for tau -> 0 and zero/undefined sigma with explicit missing flag
    """
    s_arr = np.asarray(underlying_price, dtype=float)
    k_arr = np.asarray(strike_price, dtype=float)
    sigma_arr = np.asarray(sigma_1m, dtype=float)
    tau_arr = np.asarray(tau_min, dtype=float)

    is_scalar = (
        np.ndim(underlying_price) == 0
        and np.ndim(strike_price) == 0
        and np.ndim(sigma_1m) == 0
        and np.ndim(tau_min) == 0
        and isinstance(candidate_side, str)
    )

    if isinstance(candidate_side, str):
        sides = np.full(s_arr.shape, candidate_side)
    else:
        sides = np.asarray(candidate_side)

    # Determine direction factor: +1 for UP/YES, -1 for DOWN/NO
    direction = np.where(
        np.char.upper(sides.astype(str)) == "DOWN",
        -1.0,
        np.where(np.char.upper(sides.astype(str)) == "NO", -1.0, 1.0)
    )

    # Valid mask: positive finite prices, positive finite volatility, non-negative tau
    valid_mask = (
        np.isfinite(s_arr) & (s_arr > 0.0) &
        np.isfinite(k_arr) & (k_arr > 0.0) &
        np.isfinite(sigma_arr) & (sigma_arr > 1e-9) &
        np.isfinite(tau_arr)
    )

    z = np.zeros_like(s_arr, dtype=float)

    if np.any(valid_mask):
        s_v = s_arr[valid_mask]
        k_v = k_arr[valid_mask]
        sig_v = sigma_arr[valid_mask]
        # Protect tau -> 0: effective minimum time step is 1e-4 minutes (~6ms)
        tau_eff = np.maximum(tau_arr[valid_mask], 1e-4)
        dir_v = direction[valid_mask]

        denom = sig_v * np.sqrt(tau_eff)
        # Avoid division by zero
        denom_safe = np.where(denom > 1e-9, denom, 1e-9)
        z_vals = dir_v * np.log(s_v / k_v) / denom_safe

        if clip_bounds is not None:
            z_vals = np.clip(z_vals, clip_bounds[0], clip_bounds[1])

        z[valid_mask] = z_vals

    if is_scalar:
        return float(z.item()), bool(valid_mask.item())
    return z, valid_mask


def compute_directional_momentum(
    underlying_current: float | np.ndarray,
    underlying_lagged: float | np.ndarray,
    candidate_side: str | Sequence[str],
    has_ref: bool | np.ndarray = True,
) -> tuple[np.ndarray | float, np.ndarray | bool]:
    """
    Computes directed short momentum (Item 2.3):
      ret_outsider = s * log(S_t / S_{t - delta_t})
    where s = +1 for candidate UP/YES, -1 for DOWN/NO.

    Movement towards the winning side of the outsider always yields a positive value.
    """
    s_curr = np.asarray(underlying_current, dtype=float)
    s_lag = np.asarray(underlying_lagged, dtype=float)
    ref_mask = np.asarray(has_ref, dtype=bool)

    is_scalar = (
        np.ndim(underlying_current) == 0
        and np.ndim(underlying_lagged) == 0
        and isinstance(candidate_side, str)
    )

    if isinstance(candidate_side, str):
        sides = np.full(s_curr.shape, candidate_side)
    else:
        sides = np.asarray(candidate_side)

    direction = np.where(
        np.char.upper(sides.astype(str)) == "DOWN",
        -1.0,
        np.where(np.char.upper(sides.astype(str)) == "NO", -1.0, 1.0)
    )

    valid_mask = (
        ref_mask &
        np.isfinite(s_curr) & (s_curr > 0.0) &
        np.isfinite(s_lag) & (s_lag > 0.0)
    )

    ret = np.zeros_like(s_curr, dtype=float)
    if np.any(valid_mask):
        sc_v = s_curr[valid_mask]
        sl_v = s_lag[valid_mask]
        dir_v = direction[valid_mask]
        ret[valid_mask] = dir_v * np.log(sc_v / sl_v)

    if is_scalar:
        return float(ret.item()), bool(valid_mask.item())
    return ret, valid_mask


def compute_outsider_model_features(
    df: pd.DataFrame,
    underlying_observations: Sequence[Any] | None = None,
    minute_candles: pd.DataFrame | None = None,
    sigma_window_candles: int = 60,
    z_clip_bounds: tuple[float, float] = (-5.0, 5.0),
) -> pd.DataFrame:
    """
    Computes transformed features for Model A1 and Model B1 (Items 2.2, 2.3, 2.6, 2.7).

    Features added:
    - logit_mid_price: logit(clip(outsider_mid, 1e-4, 1 - 1e-4))
    - log_time_left: log1p(clip(time_left_min, 0.0, None))
    - candidate_spread: spread of the candidate contract
    - logit_price_x_log_time: interaction term logit_mid_price * log_time_left
    - z_outsider: normalized distance to strike (Item 2.2)
    - has_z_ref: validity indicator for strike distance
    - ret_outsider_30s: 30-second directional momentum (Item 2.3)
    - has_ret_30s_ref: validity indicator for 30s momentum
    - ret_outsider_120s: 120-second directional momentum (Item 2.3)
    - has_ret_120s_ref: validity indicator for 120s momentum
    """
    out = df.copy()

    # Determine mid price
    mid = out["mid_price"] if "mid_price" in out.columns else out.get("outsider_mid", 0.5)
    mid_vals = pd.to_numeric(mid, errors="coerce").fillna(0.5).to_numpy()

    # Logit transform with numerical contract limits [1e-4, 1 - 1e-4]
    p_clip = np.clip(mid_vals, 1e-4, 1.0 - 1e-4)
    out["logit_mid_price"] = np.log(p_clip / (1.0 - p_clip))

    # Time transform log1p(time_left_min)
    time_left = out["time_left_min"] if "time_left_min" in out.columns else 15.0
    t_vals = pd.to_numeric(time_left, errors="coerce").fillna(15.0).to_numpy()
    out["log_time_left"] = np.log1p(np.maximum(t_vals, 0.0))

    # Spread
    spread = out["spread"] if "spread" in out.columns else out.get("candidate_spread", 0.02)
    out["candidate_spread"] = pd.to_numeric(spread, errors="coerce").fillna(0.02).to_numpy()

    # Interaction
    out["logit_price_x_log_time"] = out["logit_mid_price"] * out["log_time_left"]

    # Candidate side ("UP" or "DOWN", default "UP")
    candidate_sides = out.get("candidate_side", "UP")
    if isinstance(candidate_sides, pd.Series):
        sides_arr = candidate_sides.fillna("UP").astype(str).to_numpy()
    else:
        sides_arr = np.full(len(out), str(candidate_sides))

    # Strike price
    if "strike_value" in out.columns:
        strike_series = out["strike_value"]
    elif "strike_price" in out.columns:
        strike_series = out["strike_price"]
    else:
        strike_series = pd.Series(np.nan, index=out.index)
    strike_vals = pd.to_numeric(strike_series, errors="coerce").to_numpy()

    # Underlying price S_t
    und_series = out["underlying_price"] if "underlying_price" in out.columns else pd.Series(np.nan, index=out.index)
    underlying_vals = pd.to_numeric(und_series, errors="coerce").to_numpy()

    # Volatility sigma_1m
    sig_series = out["sigma_1m"] if "sigma_1m" in out.columns else pd.Series(np.nan, index=out.index)
    sigma_vals = pd.to_numeric(sig_series, errors="coerce").to_numpy()
    if np.isnan(sigma_vals).all() and minute_candles is not None and len(minute_candles) >= 10:
        # Compute rolling 1m return std over window
        close_series = pd.Series(minute_candles["close"]).astype(float)
        ret_1m = np.log(close_series / close_series.shift(1))
        # Use ddof=1 sample standard deviation
        est_sigma = float(ret_1m.tail(sigma_window_candles).std(ddof=1))
        sigma_vals = np.full(len(out), est_sigma if np.isfinite(est_sigma) else 0.0)
    elif np.isnan(sigma_vals).all():
        # Default fallback volatility (e.g. 0.001 per minute ~ 3.8% daily BTC vol)
        sigma_vals = np.full(len(out), 0.001)

    # Compute z_outsider
    z_out, z_valid = compute_normalized_strike_distance(
        underlying_price=underlying_vals,
        strike_price=strike_vals,
        sigma_1m=sigma_vals,
        tau_min=t_vals,
        candidate_side=sides_arr,
        clip_bounds=z_clip_bounds,
    )
    out["z_outsider"] = z_out
    out["has_z_ref"] = z_valid

    # Momentum features
    has_ret_30 = np.zeros(len(out), dtype=bool)
    ret_30 = np.zeros(len(out), dtype=float)
    has_ret_120 = np.zeros(len(out), dtype=bool)
    ret_120 = np.zeros(len(out), dtype=float)

    # Check if pre-computed lagged underlying prices are present
    if "underlying_lag_30s" in out.columns:
        lag30 = pd.to_numeric(out["underlying_lag_30s"], errors="coerce").to_numpy()
        ret_30, has_ret_30 = compute_directional_momentum(
            underlying_current=underlying_vals,
            underlying_lagged=lag30,
            candidate_side=sides_arr,
            has_ref=np.isfinite(lag30) & (lag30 > 0.0),
        )
    elif underlying_observations:
        from polyflip.crypto.underlying_observations import compute_underlying_return
        for idx in range(len(out)):
            t_ref = out["recorded_at"].iloc[idx] if "recorded_at" in out.columns else datetime.now(timezone.utc)
            r30, ok30 = compute_underlying_return(underlying_observations, as_of=t_ref, horizon_seconds=30.0)
            if ok30 and r30 is not None:
                s_factor = -1.0 if str(sides_arr[idx]).upper() in ("DOWN", "NO") else 1.0
                ret_30[idx] = s_factor * r30
                has_ret_30[idx] = True

    if "underlying_lag_120s" in out.columns:
        lag120 = pd.to_numeric(out["underlying_lag_120s"], errors="coerce").to_numpy()
        ret_120, has_ret_120 = compute_directional_momentum(
            underlying_current=underlying_vals,
            underlying_lagged=lag120,
            candidate_side=sides_arr,
            has_ref=np.isfinite(lag120) & (lag120 > 0.0),
        )
    elif underlying_observations:
        from polyflip.crypto.underlying_observations import compute_underlying_return
        for idx in range(len(out)):
            t_ref = out["recorded_at"].iloc[idx] if "recorded_at" in out.columns else datetime.now(timezone.utc)
            r120, ok120 = compute_underlying_return(underlying_observations, as_of=t_ref, horizon_seconds=120.0)
            if ok120 and r120 is not None:
                s_factor = -1.0 if str(sides_arr[idx]).upper() in ("DOWN", "NO") else 1.0
                ret_120[idx] = s_factor * r120
                has_ret_120[idx] = True

    out["ret_outsider_30s"] = ret_30
    out["has_ret_30s_ref"] = has_ret_30
    out["ret_outsider_120s"] = ret_120
    out["has_ret_120s_ref"] = has_ret_120

    return out

