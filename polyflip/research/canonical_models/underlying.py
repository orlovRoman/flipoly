"""Step 13: attach underlying + canonical strike in identical units.

Availability must precede the decision. Binance is kept as a separate
proxy/additional source and never silently substitutes the canonical strike.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class UnderlyingPoint:
    price: float
    unit: str  # e.g. "USD"
    available_at: datetime


@dataclass(frozen=True)
class StrikePoint:
    value: float
    unit: str
    source: str  # canonical_confirmed | retrospective | binance_proxy | unknown
    available_at: datetime | None


@dataclass(frozen=True)
class AttachedUnderlying:
    underlying: float | None
    strike: float | None
    strike_source: str
    status: str  # OK | UNIT_MISMATCH | NOT_YET_AVAILABLE | MISSING


def _utc(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def attach_underlying(up: UnderlyingPoint | None, strike: StrikePoint | None,
                     decision_at: datetime) -> AttachedUnderlying:
    dec = _utc(decision_at)
    if up is None or strike is None or strike.value is None:
        return AttachedUnderlying(None, None, getattr(strike, "source", "unknown"), "MISSING")
    if up.unit != strike.unit:
        return AttachedUnderlying(None, None, strike.source, "UNIT_MISMATCH")
    if _utc(up.available_at) > dec:
        return AttachedUnderlying(None, None, strike.source, "NOT_YET_AVAILABLE")
    if strike.available_at is not None and _utc(strike.available_at) > dec:
        return AttachedUnderlying(None, None, strike.source, "NOT_YET_AVAILABLE")
    if strike.available_at is None and strike.source in ("canonical_confirmed", "retrospective"):
        # confirmed/retrospective strike without timestamp cannot be proven
        # available at decision -> treat as not available (step 8)
        return AttachedUnderlying(None, None, strike.source, "NOT_YET_AVAILABLE")
    return AttachedUnderlying(up.price, strike.value, strike.source, "OK")
