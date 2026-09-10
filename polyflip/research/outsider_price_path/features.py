"""
polyflip/research/outsider_price_path/features.py

Causal trajectory feature extraction from pre-decision price history (Stage 3).
Strictly adheres to causality:
- Only observations at or prior to decision_at are processed.
- Subsequent trough is evaluated strictly at or after peak.
- Irregular sampling is integrated continuously (duplicate snapshots contribute zero duration).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Sequence, Optional, Any


@dataclass(frozen=True)
class Observation:
    """A single price observation timestamped with recorded_at."""
    timestamp: datetime
    mid_price: float
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryFeatures:
    """Extracted causal trajectory features for the chosen outsider token."""
    # Observations count & sampling
    n_observations: int
    span_seconds: float
    max_gap_seconds: float
    last_obs_age_seconds: float

    # Base price levels
    initial_mid: float
    current_mid: float
    max_mid: float
    min_mid: float

    # Peak-Trough-Current dynamics
    peak_mid: float
    peak_timestamp_iso: str
    subsequent_trough_mid: float
    trough_timestamp_iso: str
    drawdown: float
    rebound: float
    recovery_fraction: Optional[float]
    drop_from_peak: float
    drop_duration_sec: float
    time_peak_to_trough_sec: float
    range_position: float

    # Favorite exposure & timing
    time_above_50_sec: float
    fraction_time_above_50: float
    touch_count_above_50: int
    single_touch_above_50: bool
    time_last_above_50_sec: Optional[float]

    # Rebound indicator (pre-registered threshold: drawdown >= 0.05 & rebound >= 0.02)
    is_rebound: bool
    rebound_category: str  # "REBOUND" or "NO_REBOUND"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_trajectory_features(
    observations: Sequence[Observation],
    decision_at: datetime,
    rebound_drawdown_thresh: float = 0.05,
    rebound_rebound_thresh: float = 0.02,
) -> Optional[TrajectoryFeatures]:
    """
    Computes pure causal trajectory features from pre-decision observations.
    
    Parameters:
        observations: Sequence of Observation objects.
        decision_at: Strict decision cutoff time. Observations with timestamp > decision_at are ignored.
        rebound_drawdown_thresh: Minimum drawdown for REBOUND classification (default 0.05).
        rebound_rebound_thresh: Minimum rebound for REBOUND classification (default 0.02).
        
    Returns:
        TrajectoryFeatures or None if observations are empty after causal filtering.
    """
    # 1. Filter causally: t <= decision_at and mid_price is valid finite number
    valid_obs: list[Observation] = []
    for obs in observations:
        t = obs.timestamp
        if t.tzinfo is None and decision_at.tzinfo is not None:
            t = t.replace(tzinfo=decision_at.tzinfo)
        elif t.tzinfo is not None and decision_at.tzinfo is None:
            decision_at = decision_at.replace(tzinfo=t.tzinfo)
        
        if t <= decision_at:
            if obs.mid_price is not None and math.isfinite(obs.mid_price):
                valid_obs.append(Observation(
                    timestamp=t,
                    mid_price=float(obs.mid_price),
                    best_bid=obs.best_bid,
                    best_ask=obs.best_ask,
                    spread=obs.spread
                ))

    if not valid_obs:
        return None

    # Sort strictly ascending by timestamp
    valid_obs.sort(key=lambda o: o.timestamp)
    n_obs = len(valid_obs)

    # 2. Time metrics & sampling irregularities
    t_start = valid_obs[0].timestamp
    t_end = valid_obs[-1].timestamp
    span_sec = max(0.0, (t_end - t_start).total_seconds())
    last_obs_age_sec = max(0.0, (decision_at - t_end).total_seconds())

    max_gap_sec = 0.0
    for i in range(n_obs - 1):
        dt = (valid_obs[i + 1].timestamp - valid_obs[i].timestamp).total_seconds()
        if dt > max_gap_sec:
            max_gap_sec = dt

    # 3. Base price levels
    initial_mid = valid_obs[0].mid_price
    current_mid = valid_obs[-1].mid_price
    max_mid = max(o.mid_price for o in valid_obs)
    min_mid = min(o.mid_price for o in valid_obs)

    # Range position
    denom_range = max_mid - min_mid
    if denom_range > 1e-6:
        range_pos = max(0.0, min(1.0, (current_mid - min_mid) / denom_range))
    else:
        range_pos = 0.5

    # 4. Peak -> Subsequent Trough -> Current Price sequence
    # Find peak mid and earliest timestamp at which it occurred
    peak_mid = max_mid
    peak_idx = -1
    for i, o in enumerate(valid_obs):
        if abs(o.mid_price - peak_mid) < 1e-9:
            peak_idx = i
            break

    peak_obs = valid_obs[peak_idx]
    peak_time = peak_obs.timestamp

    # Search for subsequent trough STRICTLY at or after peak_idx
    subsequent_obs = valid_obs[peak_idx:]
    subsequent_trough_mid = min(o.mid_price for o in subsequent_obs)
    trough_idx_rel = -1
    for i, o in enumerate(subsequent_obs):
        if abs(o.mid_price - subsequent_trough_mid) < 1e-9:
            trough_idx_rel = i
            break

    trough_obs = subsequent_obs[trough_idx_rel]
    trough_time = trough_obs.timestamp

    # Drawdown, rebound, recovery_fraction
    drawdown = peak_mid - subsequent_trough_mid
    rebound = current_mid - subsequent_trough_mid
    if drawdown > 1e-6:
        recovery_fraction = max(0.0, rebound / drawdown)
    else:
        recovery_fraction = None

    drop_from_peak = peak_mid - current_mid
    drop_duration_sec = max(0.0, (t_end - peak_time).total_seconds())
    time_peak_to_trough_sec = max(0.0, (trough_time - peak_time).total_seconds())

    # 5. Continuous time spent above 0.50 (favorite status)
    # Piecewise integration over intervals [t_i, t_{i+1}]
    time_above_50 = 0.0
    for i in range(n_obs - 1):
        dt = max(0.0, (valid_obs[i + 1].timestamp - valid_obs[i].timestamp).total_seconds())
        if dt > 0.0:
            p_i = valid_obs[i].mid_price
            p_next = valid_obs[i + 1].mid_price
            # If both > 0.50, entire dt is above 0.50
            if p_i > 0.50 and p_next > 0.50:
                time_above_50 += dt
            elif p_i > 0.50 and p_next <= 0.50:
                # Linear crossing proportion
                frac = (p_i - 0.50) / (p_i - p_next)
                time_above_50 += dt * frac
            elif p_i <= 0.50 and p_next > 0.50:
                frac = (p_next - 0.50) / (p_next - p_i)
                time_above_50 += dt * frac

    fraction_time_above_50 = (time_above_50 / span_sec) if span_sec > 0.0 else (1.0 if initial_mid > 0.50 else 0.0)

    # Touch count above 0.50 and time since last observation > 0.50
    obs_above_50 = [o for o in valid_obs if o.mid_price > 0.50]
    touch_count_above_50 = len(obs_above_50)
    single_touch_above_50 = (touch_count_above_50 == 1)

    if obs_above_50:
        last_above_time = obs_above_50[-1].timestamp
        time_last_above_50_sec = max(0.0, (decision_at - last_above_time).total_seconds())
    else:
        time_last_above_50_sec = None

    # 6. Pre-registered Rebound classification
    is_rebound = (drawdown >= rebound_drawdown_thresh) and (rebound >= rebound_rebound_thresh)
    rebound_category = "REBOUND" if is_rebound else "NO_REBOUND"

    return TrajectoryFeatures(
        n_observations=n_obs,
        span_seconds=round(span_sec, 2),
        max_gap_seconds=round(max_gap_sec, 2),
        last_obs_age_seconds=round(last_obs_age_sec, 2),
        initial_mid=round(initial_mid, 6),
        current_mid=round(current_mid, 6),
        max_mid=round(max_mid, 6),
        min_mid=round(min_mid, 6),
        peak_mid=round(peak_mid, 6),
        peak_timestamp_iso=peak_time.isoformat(),
        subsequent_trough_mid=round(subsequent_trough_mid, 6),
        trough_timestamp_iso=trough_time.isoformat(),
        drawdown=round(drawdown, 6),
        rebound=round(rebound, 6),
        recovery_fraction=round(recovery_fraction, 6) if recovery_fraction is not None else None,
        drop_from_peak=round(drop_from_peak, 6),
        drop_duration_sec=round(drop_duration_sec, 2),
        time_peak_to_trough_sec=round(time_peak_to_trough_sec, 2),
        range_position=round(range_pos, 6),
        time_above_50_sec=round(time_above_50, 2),
        fraction_time_above_50=round(fraction_time_above_50, 6),
        touch_count_above_50=touch_count_above_50,
        single_touch_above_50=single_touch_above_50,
        time_last_above_50_sec=round(time_last_above_50_sec, 2) if time_last_above_50_sec is not None else None,
        is_rebound=is_rebound,
        rebound_category=rebound_category,
    )
