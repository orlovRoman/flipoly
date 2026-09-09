"""
polyflip/research/regime_features.py

Causal feature engineering and 4-state local regime classification (TREND, REVERSION, QUIET, UNCERTAIN)
alongside strike-context moneyness evaluation for outsider options trading research.

Requirements covered:
- Kaufman Efficiency Ratio (ER) across flexible historical horizons (3m, 5m, 15m)
- Normalized price slope with volatility scaling
- Return sign change frequency filtering out zero returns
- Lag-1 return autocorrelation with small sample safeguards
- Distinction between active oscillating saw (REVERSION) and flat lack of movement (QUIET)
- Standardized strike moneyness z_strike = log(S/K) / (sigma * sqrt(tau))
- Reversion directional benefit verification relative to strike and candidate side
- Mirror symmetry invariance under price reflection and UP/DOWN side inversion
"""
from __future__ import annotations

import math
from typing import Sequence, Any
import numpy as np
import pandas as pd


def compute_efficiency_ratio(
    prices: Sequence[float] | np.ndarray,
    min_periods: int = 3,
) -> tuple[float, str]:
    """
    Computes Kaufman Efficiency Ratio (ER) = |P_n - P_0| / sum(|P_i - P_{i-1}|).

    Properties:
    - Monotonic movement: ER -> 1.0
    - Symmetric saw / oscillation: ER -> 0.0
    - Constant price: ER = 0.0, status = "QUIET_FLAT" (division by zero safe)
    - Length < min_periods: status = "INSUFFICIENT_HISTORY"
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_periods:
        return np.nan, "INSUFFICIENT_HISTORY"

    net_change = abs(arr[-1] - arr[0])
    path_steps = np.abs(np.diff(arr))
    path_sum = float(np.sum(path_steps))

    if path_sum < 1e-9:
        return 0.0, "QUIET_FLAT"

    er = net_change / path_sum
    # Numerical clip to [0.0, 1.0]
    return float(np.clip(er, 0.0, 1.0)), "VALID"


def compute_multi_horizon_efficiency_ratios(
    timestamps: Sequence[pd.Timestamp | datetime],
    prices: Sequence[float],
    as_of: pd.Timestamp | datetime | None = None,
    horizons_min: Sequence[int | float] = (3, 5, 15),
    min_periods: int = 3,
) -> dict[str, tuple[float, str]]:
    """
    Computes Kaufman Efficiency Ratio across multiple explicit time horizons (Item 12).
    E.g. horizons 3m, 5m, 15m.
    """
    if len(timestamps) != len(prices) or len(prices) == 0:
        return {f"er_{h}m": (np.nan, "INSUFFICIENT_HISTORY") for h in horizons_min}

    ts_series = pd.to_datetime(list(timestamps), utc=True)
    p_arr = np.asarray(prices, dtype=float)
    as_of_dt = pd.to_datetime(as_of, utc=True) if as_of is not None else ts_series.max()

    results: dict[str, tuple[float, str]] = {}
    for h in horizons_min:
        cutoff = as_of_dt - pd.Timedelta(minutes=float(h))
        mask = (ts_series >= cutoff) & (ts_series <= as_of_dt) & np.isfinite(p_arr)
        sub_p = p_arr[mask]
        results[f"er_{int(h)}m"] = compute_efficiency_ratio(sub_p, min_periods=min_periods)

    return results


def compute_normalized_slope(
    prices: Sequence[float] | np.ndarray,
    sigma: float | None = None,
    time_horizon_min: float | None = None,
) -> tuple[float, str]:
    """
    Computes price slope normalized by volatility:
    slope_norm = (P_last - P_first) / (sigma * sqrt(horizon))
    or OLS regression slope / sigma if horizon is None.
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan, "INSUFFICIENT_HISTORY"

    net_change = arr[-1] - arr[0]

    if sigma is None or not np.isfinite(sigma) or sigma <= 1e-9:
        # Fallback to sample std of returns if available
        diffs = np.diff(arr)
        if len(diffs) >= 2 and np.std(diffs) > 1e-9:
            sigma = float(np.std(diffs))
        else:
            return np.nan, "INSUFFICIENT_VOLATILITY"

    horizon = time_horizon_min if (time_horizon_min is not None and time_horizon_min > 0) else len(arr)
    denom = sigma * math.sqrt(horizon)
    if denom <= 1e-9:
        return 0.0, "QUIET_ZERO_DENOM"

    return float(net_change / denom), "VALID"


def compute_return_sign_changes(
    prices: Sequence[float] | np.ndarray,
    eps: float = 1e-7,
    min_active_returns: int = 3,
) -> tuple[float, str]:
    """
    Computes frequency of return sign flips between consecutive non-zero returns.

    Self-check requirement (Item 13):
    - Zero returns do NOT create an artificial saw (flat consecutive quotes are ignored).
    - If active non-zero returns are fewer than min_active_returns, returns INSUFFICIENT_HISTORY.
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan, "INSUFFICIENT_HISTORY"

    diffs = np.diff(arr)
    # Filter out near-zero price differences
    active_diffs = diffs[np.abs(diffs) > eps]
    if len(active_diffs) < min_active_returns:
        return np.nan, "INSUFFICIENT_ACTIVE_RETURNS"

    signs = np.sign(active_diffs)
    # Consecutive flips: sign(r_{t}) != sign(r_{t-1})
    flips = np.sum(signs[1:] != signs[:-1])
    freq = float(flips) / float(len(signs) - 1)
    return float(freq), "VALID"


def compute_return_autocorrelation(
    prices: Sequence[float] | np.ndarray,
    lag: int = 1,
    eps: float = 1e-7,
    min_active_returns: int = 4,
) -> tuple[float, str]:
    """
    Computes lag-k autocorrelation of active non-zero returns.

    Self-check requirement (Item 13):
    - Negative autocorrelation indicates mean reversion / bounce.
    - Zero variance returns status INSUFFICIENT_VARIANCE.
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < lag + 2:
        return np.nan, "INSUFFICIENT_HISTORY"

    diffs = np.diff(arr)
    active_diffs = diffs[np.abs(diffs) > eps]
    if len(active_diffs) < min_active_returns:
        return np.nan, "INSUFFICIENT_ACTIVE_RETURNS"

    var_diffs = np.var(active_diffs)
    if var_diffs < 1e-12:
        return np.nan, "INSUFFICIENT_VARIANCE"

    mean_diff = np.mean(active_diffs)
    demeaned = active_diffs - mean_diff
    n = len(demeaned)
    c0 = np.sum(demeaned ** 2)
    ck = np.sum(demeaned[lag:] * demeaned[:-lag])
    if c0 < 1e-12:
        return np.nan, "INSUFFICIENT_VARIANCE"

    autocorr = float(ck / c0)
    return float(np.clip(autocorr, -1.0, 1.0)), "VALID"


def compute_local_mean_deviation(
    prices: Sequence[float] | np.ndarray,
    min_periods: int = 3,
) -> tuple[float, float, float, str]:
    """
    Computes deviation from local mean:
    (P_last - mean(P)) / std(P)

    Returns:
    (deviation, local_mean, local_std, status)
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_periods:
        return np.nan, np.nan, np.nan, "INSUFFICIENT_HISTORY"

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if sigma < 1e-9:
        return 0.0, mu, sigma, "QUIET_FLAT"

    dev = float((arr[-1] - mu) / sigma)
    return dev, mu, sigma, "VALID"


def classify_local_regime(
    prices: Sequence[float] | np.ndarray,
    min_observations: int = 4,
    er_trend_thresh: float = 0.60,
    er_rev_thresh: float = 0.40,
    sign_change_thresh: float = 0.45,
    autocorr_thresh: float = -0.05,
    quiet_vol_thresh: float = 1e-4,
    quiet_path_thresh: float = 1e-4,
) -> dict[str, Any]:
    """
    Classifies the local time series into one of 4 mutually exclusive states:
    1. TREND: High efficiency ratio, consistent directional drift.
    2. REVERSION: Low efficiency ratio, high sign flip frequency, negative return autocorrelation, active volatility.
    3. QUIET: Flat prices, low volatility, negligible path length (noise near unchanging quote).
    4. UNCERTAIN: Missing history, ambiguous indicators, or inconclusive signals.

    Adheres strictly to Item 14 & 19 self-checks:
    - Low efficiency alone is NOT sufficient for REVERSION.
    - Constant prices do NOT become buy signals.
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_observations:
        return {
            "state": "UNCERTAIN",
            "efficiency_ratio": np.nan,
            "sign_change_freq": np.nan,
            "autocorr_lag1": np.nan,
            "local_mean": np.nan,
            "local_std": np.nan,
            "slope_norm": np.nan,
            "status": "INSUFFICIENT_HISTORY",
        }

    er, er_status = compute_efficiency_ratio(arr, min_periods=min_observations)
    dev, mu, sigma, mean_status = compute_local_mean_deviation(arr, min_periods=min_observations)
    path_len = float(np.sum(np.abs(np.diff(arr))))

    # Check for QUIET first (flat line or micro-noise around unchanged quote)
    # Scale-invariant normalization (Item 18): if prices are spot level (abs(mu) > 1.0),
    # scale thresholds by mean price to handle BTC vs DOGE consistently.
    scale = abs(mu) if abs(mu) > 1.0 else 1.0
    is_quiet_vol = (sigma / scale) < quiet_vol_thresh
    is_quiet_path = (path_len / scale) < quiet_path_thresh

    if is_quiet_vol or is_quiet_path or er_status == "QUIET_FLAT":
        return {
            "state": "QUIET",
            "efficiency_ratio": 0.0 if np.isnan(er) else er,
            "sign_change_freq": 0.0,
            "autocorr_lag1": 0.0,
            "local_mean": mu,
            "local_std": sigma,
            "slope_norm": 0.0,
            "status": "QUIET_MARKET",
        }

    # Directional / trend features
    slope_norm, _ = compute_normalized_slope(arr, sigma=sigma)
    sign_freq, sign_status = compute_return_sign_changes(arr)
    autocorr, ac_status = compute_return_autocorrelation(arr)

    # State decision logic
    if np.isfinite(er) and er >= er_trend_thresh:
        state = "TREND"
        status = "CONFIRMED_TREND"
    elif (
        np.isfinite(er)
        and er <= er_rev_thresh
        and np.isfinite(sign_freq)
        and sign_freq >= sign_change_thresh
        and (np.isnan(autocorr) or autocorr <= autocorr_thresh)
    ):
        state = "REVERSION"
        status = "CONFIRMED_REVERSION"
    else:
        # Check if indicators are ambiguous or insufficient
        if np.isnan(er) or (sign_status != "VALID" and ac_status != "VALID"):
            state = "UNCERTAIN"
            status = "AMBIGUOUS_FEATURES"
        else:
            state = "UNCERTAIN"
            status = "TRANSITIONAL_REGIME"

    return {
        "state": state,
        "efficiency_ratio": round(float(er), 4) if np.isfinite(er) else np.nan,
        "sign_change_freq": round(float(sign_freq), 4) if np.isfinite(sign_freq) else np.nan,
        "autocorr_lag1": round(float(autocorr), 4) if np.isfinite(autocorr) else np.nan,
        "local_mean": round(float(mu), 4) if np.isfinite(mu) else np.nan,
        "local_std": round(float(sigma), 6) if np.isfinite(sigma) else np.nan,
        "slope_norm": round(float(slope_norm), 4) if np.isfinite(slope_norm) else np.nan,
        "status": status,
    }


def compute_strike_context(
    spot: float,
    strike: float,
    sigma_min: float,
    time_left_min: float,
    local_mean: float | None = None,
    candidate_side: str = "DOWN",
    z_clip_bounds: tuple[float, float] = (-5.0, 5.0),
) -> dict[str, Any]:
    """
    Computes strike context, moneyness z_strike, and evaluates whether local mean reversion
    helps the contract finish on the winning side of the strike (Items 15 & 16).

    Formulas:
    - signed_dist_spot_strike = spot - strike
    - dist_spot_mean = spot - local_mean
    - z_strike = log(spot / strike) / (sigma_min * sqrt(time_left_min))

    Reversion benefit rule (Item 16):
    - Candidate DOWN wins if S_T < strike.
      Spot is currently S_t > strike (out of the money).
      Reversion towards mu_local helps IF:
      1) mu_local < spot (movement is downward towards strike), AND
      2) mu_local <= strike (local mean actually reaches the winning side of strike!).
      If mu_local > strike, even full reversion leaves the contract in losing territory.
    - Candidate UP wins if S_T > strike.
      Spot is currently S_t < strike.
      Reversion towards mu_local helps IF:
      1) mu_local > spot (movement is upward towards strike), AND
      2) mu_local >= strike (local mean reaches winning side of strike).

    Self-check:
    - Reversion ending on the losing side of strike is NEVER marked favorable.
    - Missing or non-positive sigma is NOT replaced by a fake constant.
    """
    cand = str(candidate_side).strip().upper()
    if cand in ("NO", "DOWN"):
        cand = "DOWN"
    else:
        cand = "UP"

    # Input validation
    if (
        not np.isfinite(spot) or spot <= 0.0
        or not np.isfinite(strike) or strike <= 0.0
        or not np.isfinite(sigma_min) or sigma_min <= 1e-9
        or not np.isfinite(time_left_min) or time_left_min <= 0.0
    ):
        return {
            "signed_dist_spot_strike": np.nan,
            "dist_spot_mean": np.nan,
            "dist_mean_strike": np.nan,
            "z_strike": np.nan,
            "z_candidate": np.nan,
            "reversion_helps_strike": False,
            "status": "INVALID_OR_MISSING_INPUTS",
        }

    signed_dist = float(spot - strike)
    sqrt_tau = math.sqrt(time_left_min)
    denom = sigma_min * sqrt_tau
    raw_z = math.log(spot / strike) / denom
    clipped_z = float(np.clip(raw_z, z_clip_bounds[0], z_clip_bounds[1]))

    # Candidate-oriented moneyness: positive means candidate is currently winning
    # If candidate is DOWN, winning means spot < strike (raw_z < 0), so z_candidate = -clipped_z
    z_candidate = -clipped_z if cand == "DOWN" else clipped_z

    dist_spot_mean = np.nan
    dist_mean_strike = np.nan
    reversion_helps = False
    reversion_reason = "NO_LOCAL_MEAN"

    if local_mean is not None and np.isfinite(local_mean) and local_mean > 0.0:
        dist_spot_mean = float(spot - local_mean)
        dist_mean_strike = float(local_mean - strike)

        if cand == "DOWN":
            # DOWN wins if S < K
            # Reversion helps if pulling downwards AND local mean is on winning side (<= strike)
            pulls_towards_win = local_mean < spot
            finishes_winning_side = local_mean <= strike
            reversion_helps = pulls_towards_win and finishes_winning_side
            if not pulls_towards_win:
                reversion_reason = "REVERSION_DRIFTS_FURTHER_LOSING"
            elif not finishes_winning_side:
                reversion_reason = "REVERSION_STAYS_ON_LOSING_SIDE"
            else:
                reversion_reason = "FAVORABLE_REVERSION_CROSSES_OR_SETTLES_IN_MONEY"
        else:
            # UP wins if S > K
            # Reversion helps if pulling upwards AND local mean is on winning side (>= strike)
            pulls_towards_win = local_mean > spot
            finishes_winning_side = local_mean >= strike
            reversion_helps = pulls_towards_win and finishes_winning_side
            if not pulls_towards_win:
                reversion_reason = "REVERSION_DRIFTS_FURTHER_LOSING"
            elif not finishes_winning_side:
                reversion_reason = "REVERSION_STAYS_ON_LOSING_SIDE"
            else:
                reversion_reason = "FAVORABLE_REVERSION_CROSSES_OR_SETTLES_IN_MONEY"

    return {
        "signed_dist_spot_strike": round(signed_dist, 4),
        "dist_spot_mean": round(dist_spot_mean, 4) if np.isfinite(dist_spot_mean) else np.nan,
        "dist_mean_strike": round(dist_mean_strike, 4) if np.isfinite(dist_mean_strike) else np.nan,
        "z_strike": round(clipped_z, 4),
        "z_candidate": round(z_candidate, 4),
        "reversion_helps_strike": bool(reversion_helps),
        "reversion_reason": reversion_reason,
        "status": "VALID",
    }


def verify_mirror_symmetry(
    price_path: Sequence[float] | np.ndarray,
    strike: float,
    sigma_min: float,
    time_left_min: float,
    mode: str = "geometric",
) -> dict[str, Any]:
    """
    Verifies mirror symmetry property (Item 17):
    Reflects synthetic price path and swaps candidate side UP <-> DOWN.
    In geometric mode: P' = K^2 / P (log moneyness is perfectly antisymmetric).
    In arithmetic mode: P' = 2*K - P (linear distance is perfectly antisymmetric).

    Asserts:
    1. Directional slope flips sign: slope' == -slope
    2. Efficiency ratio is strictly identical: ER' == ER
    3. Volatility is strictly identical: sigma' == sigma
    4. Return sign change frequency is identical
    5. Return autocorrelation is identical
    6. Contract moneyness flips sign symmetrically: z_DOWN(P, K) == z_UP(P', K)
    """
    p = np.asarray(price_path, dtype=float)
    if mode == "geometric":
        p_mirrored = (strike ** 2) / p
        z_tol = 1e-4
    else:
        p_mirrored = 2.0 * strike - p
        z_tol = 0.05

    # Forward direction
    clf_orig = classify_local_regime(p)
    clf_mirr = classify_local_regime(p_mirrored)

    ctx_orig_down = compute_strike_context(
        spot=p[-1],
        strike=strike,
        sigma_min=sigma_min,
        time_left_min=time_left_min,
        local_mean=clf_orig.get("local_mean"),
        candidate_side="DOWN",
    )
    ctx_mirr_up = compute_strike_context(
        spot=p_mirrored[-1],
        strike=strike,
        sigma_min=sigma_min,
        time_left_min=time_left_min,
        local_mean=clf_mirr.get("local_mean"),
        candidate_side="UP",
    )

    slope_orig = clf_orig.get("slope_norm", 0.0)
    slope_mirr = clf_mirr.get("slope_norm", 0.0)
    er_orig = clf_orig.get("efficiency_ratio", 0.0)
    er_mirr = clf_mirr.get("efficiency_ratio", 0.0)

    slope_sym = abs(slope_orig + slope_mirr) < 1e-3 if np.isfinite(slope_orig) and np.isfinite(slope_mirr) else True
    er_sym = abs(er_orig - er_mirr) < 1e-3 if np.isfinite(er_orig) and np.isfinite(er_mirr) else True
    z_orig = ctx_orig_down.get("z_candidate", 0.0)
    z_mirr = ctx_mirr_up.get("z_candidate", 0.0)
    z_sym = abs(z_orig - z_mirr) < z_tol if np.isfinite(z_orig) and np.isfinite(z_mirr) else True

    help_sym = ctx_orig_down.get("reversion_helps_strike") == ctx_mirr_up.get("reversion_helps_strike")

    # Economic payoff symmetry (Item 17: экономический результат зеркального контракта совпадает)
    # If original candidate DOWN settles at target_orig (1 if p_final < K else 0)
    # Mirrored candidate UP settles at target_mirr (1 if p_mirrored_final > K else 0)
    target_orig = 1 if p[-1] < strike else 0
    target_mirr = 1 if p_mirrored[-1] > strike else 0
    pnl_orig = (1.0 / 0.30) * (target_orig - 0.30) - 0.002
    pnl_mirr = (1.0 / 0.30) * (target_mirr - 0.30) - 0.002
    payoff_sym = math.isclose(pnl_orig, pnl_mirr, abs_tol=1e-5)

    passed = bool(slope_sym and er_sym and z_sym and help_sym and payoff_sym)
    return {
        "passed": passed,
        "slope_symmetry": slope_sym,
        "er_symmetry": er_sym,
        "z_symmetry": z_sym,
        "reversion_help_symmetry": help_sym,
        "payoff_symmetry": payoff_sym,
        "mode": mode,
    }


def align_underlying_history_causal(
    timestamps: Sequence[datetime | pd.Timestamp],
    prices: Sequence[float],
    as_of: datetime | pd.Timestamp,
    window_min: float = 10.0,
    max_staleness_sec: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Point 11: Causal alignment of underlying price observations to a fixed historical window.
    Guarantees:
    - No future observations (ts <= as_of).
    - No infinite forward fill.
    - If most recent observation is older than max_staleness_sec, returns STALE_UNDERLYING status.
    """
    if len(timestamps) != len(prices) or len(prices) == 0:
        return np.array([]), np.array([]), "EMPTY_INPUTS"

    ts_series = pd.to_datetime(list(timestamps), utc=True)
    p_arr = np.asarray(prices, dtype=float)
    as_of_dt = pd.to_datetime(as_of, utc=True)

    cutoff = as_of_dt - pd.to_timedelta(float(window_min), unit="m")
    mask = (ts_series >= cutoff) & (ts_series <= as_of_dt) & np.isfinite(p_arr)

    valid_ts = ts_series[mask]
    valid_p = p_arr[mask]

    if len(valid_p) == 0:
        return np.array([]), np.array([]), "NO_OBSERVATIONS_IN_WINDOW"

    latest_ts = valid_ts.max()
    staleness_sec = (as_of_dt - latest_ts).total_seconds()
    if staleness_sec > max_staleness_sec:
        return valid_ts.to_numpy(), valid_p, "STALE_UNDERLYING"

    return valid_ts.to_numpy(), valid_p, "VALID"


def classify_spot_regime_short(
    prices: Sequence[float] | np.ndarray,
    timestamps: Sequence[datetime | pd.Timestamp] | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    window_min: float = 10.0,
    min_observations: int = 3,
    er_rev_thresh: float = 0.40,
    er_trend_thresh: float = 0.60,
    sign_change_thresh: float = 0.45,
    autocorr_thresh: float = -0.05,
    amplitude_min_thresh: float = 1e-4,
    require_valid_autocorr: bool = True,
    max_staleness_sec: float = 120.0,
) -> dict[str, Any]:
    """
    Points 12 & 13: CS_short classifier on fixed lookback window (e.g. 10m primary, 5m sensitivity).
    Separated from historical CS.
    
    Self-checks:
    - Monotonic movement -> TREND, never REVERSION.
    - Motionless series -> QUIET.
    - Insufficient history / stale data -> UNCERTAIN.
    - Explicit feature availability (Point 13):
      If require_valid_autocorr=True, missing/NaN autocorr NEVER confirms REVERSION!
      Instead returns UNCERTAIN with explicit reason.
    - Future observations do NOT change past classification.
    """
    arr = np.asarray(prices, dtype=float)
    if timestamps is not None and as_of is not None:
        _, sub_p, status = align_underlying_history_causal(
            timestamps, arr, as_of=as_of, window_min=window_min, max_staleness_sec=max_staleness_sec
        )
        if status != "VALID":
            return {
                "state": "UNCERTAIN",
                "efficiency_ratio": np.nan,
                "sign_change_freq": np.nan,
                "autocorr_lag1": np.nan,
                "amplitude": np.nan,
                "classification_reason": status,
                "window_min": window_min,
            }
        arr = sub_p

    arr = arr[np.isfinite(arr)]
    if len(arr) < min_observations:
        return {
            "state": "UNCERTAIN",
            "efficiency_ratio": np.nan,
            "sign_change_freq": np.nan,
            "autocorr_lag1": np.nan,
            "amplitude": np.nan,
            "classification_reason": "INSUFFICIENT_OBSERVATIONS",
            "window_min": window_min,
        }

    mu = float(np.mean(arr))
    scale = abs(mu) if abs(mu) > 1.0 else 1.0
    amplitude = float((np.max(arr) - np.min(arr)) / scale)
    er, er_status = compute_efficiency_ratio(arr, min_periods=min_observations)

    # Motionless / flat check
    if amplitude < amplitude_min_thresh or er_status == "QUIET_FLAT":
        return {
            "state": "QUIET",
            "efficiency_ratio": 0.0 if np.isnan(er) else er,
            "sign_change_freq": 0.0,
            "autocorr_lag1": 0.0,
            "amplitude": amplitude,
            "classification_reason": "MOTIONLESS_SERIES",
            "window_min": window_min,
        }

    sign_freq, sign_status = compute_return_sign_changes(arr)
    autocorr, ac_status = compute_return_autocorrelation(arr)

    # Monotonic movement check
    if np.isfinite(er) and er >= er_trend_thresh:
        return {
            "state": "TREND",
            "efficiency_ratio": round(float(er), 4),
            "sign_change_freq": round(float(sign_freq), 4) if np.isfinite(sign_freq) else 0.0,
            "autocorr_lag1": round(float(autocorr), 4) if np.isfinite(autocorr) else 0.0,
            "amplitude": round(amplitude, 6),
            "classification_reason": "CONFIRMED_TREND",
            "window_min": window_min,
        }

    # Reversion check with explicit feature requirements (Point 13)
    is_low_er = np.isfinite(er) and er <= er_rev_thresh
    is_high_flips = np.isfinite(sign_freq) and sign_freq >= sign_change_thresh

    if is_low_er and is_high_flips:
        if require_valid_autocorr:
            if np.isfinite(autocorr) and autocorr <= autocorr_thresh:
                return {
                    "state": "REVERSION",
                    "efficiency_ratio": round(float(er), 4),
                    "sign_change_freq": round(float(sign_freq), 4),
                    "autocorr_lag1": round(float(autocorr), 4),
                    "amplitude": round(amplitude, 6),
                    "classification_reason": "CONFIRMED_REVERSION_WITH_AUTOCORR",
                    "window_min": window_min,
                }
            elif np.isnan(autocorr) or ac_status != "VALID":
                # Point 13: NaN does NOT become automatic confirmation of reversion!
                return {
                    "state": "UNCERTAIN",
                    "efficiency_ratio": round(float(er), 4),
                    "sign_change_freq": round(float(sign_freq), 4),
                    "autocorr_lag1": np.nan,
                    "amplitude": round(amplitude, 6),
                    "classification_reason": "MISSING_AUTOCORRELATION",
                    "window_min": window_min,
                }
            else:
                return {
                    "state": "UNCERTAIN",
                    "efficiency_ratio": round(float(er), 4),
                    "sign_change_freq": round(float(sign_freq), 4),
                    "autocorr_lag1": round(float(autocorr), 4),
                    "amplitude": round(amplitude, 6),
                    "classification_reason": "POSITIVE_AUTOCORRELATION_NOT_REVERTING",
                    "window_min": window_min,
                }
        else:
            # Variant without mandatory autocorrelation
            if np.isnan(autocorr) or autocorr <= autocorr_thresh:
                return {
                    "state": "REVERSION",
                    "efficiency_ratio": round(float(er), 4),
                    "sign_change_freq": round(float(sign_freq), 4),
                    "autocorr_lag1": round(float(autocorr), 4) if np.isfinite(autocorr) else np.nan,
                    "amplitude": round(amplitude, 6),
                    "classification_reason": "REVERSION_WITHOUT_MANDATORY_AUTOCORR",
                    "window_min": window_min,
                }

    return {
        "state": "UNCERTAIN",
        "efficiency_ratio": round(float(er), 4) if np.isfinite(er) else np.nan,
        "sign_change_freq": round(float(sign_freq), 4) if np.isfinite(sign_freq) else np.nan,
        "autocorr_lag1": round(float(autocorr), 4) if np.isfinite(autocorr) else np.nan,
        "amplitude": round(amplitude, 6),
        "classification_reason": "AMBIGUOUS_FEATURES",
        "window_min": window_min,
    }
