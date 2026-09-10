"""Step 15: base features log_distance / time_left / z.

Units: sigma is per-second (see volatility.py), time_left in seconds ->
z is dimensionless. Zero volatility / missing data never yields inf:
status carries ZERO_VOL or MISSING instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class BaseFeatures:
    log_distance: float | None
    time_left_sec: float | None
    sigma_per_sec: float | None
    z: float | None
    status: str  # OK | MISSING | ZERO_VOL | BAD_INPUT


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def base_features(underlying: float | None, strike: float | None,
                  sigma_per_sec: float | None,
                  decision_at: datetime, end_at: datetime) -> BaseFeatures:
    if underlying is None or strike is None or sigma_per_sec is None:
        return BaseFeatures(None, None, sigma_per_sec, None, "MISSING")
    if underlying <= 0 or strike <= 0:
        return BaseFeatures(None, None, sigma_per_sec, None, "BAD_INPUT")
    dec, end = _utc(decision_at), _utc(end_at)
    tl = (end - dec).total_seconds()
    if tl <= 0:
        return BaseFeatures(None, tl, sigma_per_sec, None, "BAD_INPUT")
    ld = math.log(underlying / strike)
    if sigma_per_sec <= 0:
        return BaseFeatures(ld, tl, sigma_per_sec, None, "ZERO_VOL")
    z = ld / (sigma_per_sec * math.sqrt(tl))
    if not math.isfinite(z):
        return BaseFeatures(ld, tl, sigma_per_sec, None, "BAD_INPUT")
    return BaseFeatures(ld, tl, sigma_per_sec, z, "OK")
