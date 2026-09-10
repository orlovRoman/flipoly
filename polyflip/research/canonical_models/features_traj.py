"""Step 16: small fixed trajectory feature set (formulas + synthetic tests).

All inputs are (t_seconds_before_decision, price) with t <= 0 and the
decision at t=0. 'Recovery' uses only the minimum strictly before decision.

Features:
- ret_1m/ret_3m/ret_5m: log(p0/past)
- n_cross_5m: strike crossings in last 5m (sign changes of p-strike)
- secs_since_cross: time since last crossing (None if no crossing)
- d_norm_1m: change of (p-strike)/strike over last 1m
- range_pos_5m: (p0-min5)/(max5-min5), None if flat
- bounce_from_min: (p0-min5)/min5 ; pullback_from_max: (max5-p0)/max5
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajFeatures:
    ret_1m: float | None
    ret_3m: float | None
    ret_5m: float | None
    n_cross_5m: int
    secs_since_cross: float | None
    d_norm_1m: float | None
    range_pos_5m: float | None
    bounce_from_min: float | None
    pullback_from_max: float | None


TRAJ_COLUMNS = (
    "ret_1m", "ret_3m", "ret_5m", "n_cross_5m", "secs_since_cross",
    "d_norm_1m", "range_pos_5m", "bounce_from_min", "pullback_from_max",
)


def _price_at(series, t_target: float) -> float | None:
    """Last price with t <= t_target (series sorted by t asc, t<=0)."""
    best = None
    for t, p in series:
        if t <= t_target:
            best = p
        else:
            break
    return best


def trajectory_features(series: list[tuple[float, float]], strike: float) -> TrajFeatures:
    s = sorted(series, key=lambda x: x[0])
    s = [(t, p) for t, p in s if t <= 0]
    if not s or strike is None or strike <= 0:
        return TrajFeatures(None, None, None, 0, None, None, None, None, None)
    p0 = s[-1][1]

    def lret(t_back: float):
        past = _price_at(s, -t_back)
        if past is None or past <= 0 or p0 <= 0:
            return None
        return math.log(p0 / past)

    w5 = [(t, p) for t, p in s if t >= -300]
    # Ignore exact touches (p == strike): compress to non-zero signs so a
    # -1 -> 0 -> +1 ramp counts as one crossing, not zero.
    nz = [((t, 1 if p > strike else -1)) for t, p in w5 if p != strike]
    n_cross, last_cross_t = 0, None
    for i in range(1, len(nz)):
        if nz[i][1] != nz[i - 1][1]:
            n_cross += 1
            last_cross_t = nz[i][0]
    secs_since = (0.0 - last_cross_t) if last_cross_t is not None else None

    p_1m = _price_at(s, -60.0)
    if p_1m and p_1m > 0:
        d_now = (p0 - strike) / strike
        d_then = (p_1m - strike) / strike
        d_norm = d_now - d_then
    else:
        d_norm = None

    prices5 = [p for _, p in w5]
    if prices5:
        mn, mx = min(prices5), max(prices5)
        rng_pos = (p0 - mn) / (mx - mn) if mx > mn else None
        bounce = (p0 - mn) / mn if mn > 0 else None
        pullback = (mx - p0) / mx if mx > 0 else None
    else:
        rng_pos, bounce, pullback = None, None, None

    return TrajFeatures(
        lret(60.0), lret(180.0), lret(300.0),
        n_cross, secs_since, d_norm, rng_pos, bounce, pullback,
    )


def to_row(f: TrajFeatures) -> dict:
    return {
        "ret_1m": f.ret_1m, "ret_3m": f.ret_3m, "ret_5m": f.ret_5m,
        "n_cross_5m": float(f.n_cross_5m),
        "secs_since_cross": f.secs_since_cross,
        "d_norm_1m": f.d_norm_1m, "range_pos_5m": f.range_pos_5m,
        "bounce_from_min": f.bounce_from_min,
        "pullback_from_max": f.pullback_from_max,
    }
