"""Step 11: causal decision-point selection at T-5m.

Rule: target = end_at - 5min. Take the FIRST available observation with
event_at in (T-5m, T-4m45s], i.e. 0 < (event_at - target) <= 15s.
If none -> MISS (skip), never fall back to a later/earlier quote.
Selection must not depend on future outcome or price advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class Observation:
    event_at: datetime
    received_at: datetime
    payload: dict | None = None


@dataclass(frozen=True)
class Decision:
    decision_at: datetime
    observation: Observation
    lateness_sec: float


TARGET_OFFSET = timedelta(minutes=5)
MAX_LATENESS = timedelta(seconds=15)


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def select_decision(observations: list[Observation], end_at: datetime):
    """Return Decision or None (MISS). Pure function of past availability."""
    end = _utc(end_at)
    target = end - TARGET_OFFSET
    window_end = target + MAX_LATENESS
    cands = []
    for o in observations:
        ev = _utc(o.event_at)
        rx = _utc(o.received_at)
        # must have been available: received no later than its event + lateness
        # and event inside (target, window_end]
        if ev <= target or ev > window_end:
            continue
        if rx > ev:
            # received after event: only usable if still received within window
            if rx > window_end:
                continue
        cands.append((ev, o))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    ev, obs = cands[0]
    return Decision(decision_at=ev, observation=obs, lateness_sec=(ev - target).total_seconds())
