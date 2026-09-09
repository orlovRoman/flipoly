"""Feature computation for strike-baseline research (causal, protocol P1-08..P1-16).

Volatility: rolling std of 1m log returns over vol_est_window_min PAST CLOSED
bars only. Needs >= vol_est_min_bars closed bars or status becomes
vol_insufficient. Never invented: no imputation, no minimum-artificial sigma.

z = log_moneyness / (sigma_min * sqrt(time_left_min))
p0 = Phi(z) is a FEATURE/control only; NEVER uses outcomes (P1-16).

Time units: time_left in minutes; sigma in per-minute units. Seconds handled by
conversion through minutes so the minutes-vs-seconds rescaling check (P1-11) holds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

STATUS_FEATURES_OK = "ok"
STATUS_NO_UNDERLYING = "no_underlying_at_decision"
STATUS_VOL_INSUFFICIENT = "vol_insufficient"
STATUS_VOL_ZERO = "vol_zero"
STATUS_Z_INFINITE = "z_infinite"


@dataclass
class FeatureResult:
    status: str
    sigma_min: float | None
    z: float | None
    p0: float | None
    n_vol_bars: int | None


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_features(
    underlying_at_decision: float | None,
    strike: float | None,
    time_left_min: float,
    returns: pd.Series | None,
) -> FeatureResult:
    """Compute sigma, z and p0 for one opportunity.

    returns: 1m log returns of the underlying for CLOSED bars strictly before
    the decision moment (the rolling window content). time_left_min > 0.

    Statuses follow P1-08 (future ticks never change features) and P1-13
    (constant series / short history / long gap never create infinite z or
    100% confidence).
    """
    if underlying_at_decision is None or strike is None or strike <= 0:
        return FeatureResult(
            STATUS_NO_UNDERLYING, None, None, None, None,
        )
    n = 0
    sigma: float | None = None
    if returns is not None and returns.size > 0:
        arr = returns.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        if n > 0:
            sigma = float(arr.std(ddof=1))
    if n < 30:
        return FeatureResult(STATUS_VOL_INSUFFICIENT, None, None, None, n)
    if sigma is not None and not math.isfinite(sigma):
        return FeatureResult(STATUS_VOL_INSUFFICIENT, None, None, None, n)
    if sigma is None or sigma <= 0:
        return FeatureResult(STATUS_VOL_ZERO, None, None, None, n)

    moneyness = math.log(underlying_at_decision / strike)
    if time_left_min is None or not math.isfinite(time_left_min) or time_left_min <= 0:
        return FeatureResult(STATUS_Z_INFINITE, sigma, None, None, n)
    z = moneyness / (sigma * math.sqrt(time_left_min))
    if not math.isfinite(z):
        return FeatureResult(STATUS_Z_INFINITE, sigma, None, None, n)
    p0 = _phi(z)
    return FeatureResult(STATUS_FEATURES_OK, sigma, z, p0, n)