"""Step 12: attach quotes of both sides using only snapshots known at decision.

- Future snapshots (event_at > decision_at) are excluded.
- Unknown time is never replaced with zero age.
- NO is never restored via 1-YES: both sides must be observed; otherwise
  the side is marked missing and the market is skipped for trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class QuoteSnapshot:
    token_id: str
    side: str  # "UP" | "DOWN"
    bid: float
    ask: float
    event_at: datetime | None
    received_at: datetime | None


@dataclass(frozen=True)
class AttachedQuotes:
    up_bid: float | None
    up_ask: float | None
    down_bid: float | None
    down_ask: float | None
    up_mid: float | None
    quote_age_sec: float | None
    status: str  # OK | MISSING_SIDE | NO_SNAPSHOT | UNKNOWN_TIME


def _utc(dt):
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def attach_quotes(snapshots: list[QuoteSnapshot], decision_at: datetime) -> AttachedQuotes:
    dec = _utc(decision_at)
    usable = [s for s in snapshots if s.event_at is not None and _utc(s.event_at) <= dec]
    if not usable:
        # distinguish unknown-time vs truly absent
        if any(s.event_at is None for s in snapshots):
            return AttachedQuotes(None, None, None, None, None, None, "UNKNOWN_TIME")
        return AttachedQuotes(None, None, None, None, None, None, "NO_SNAPSHOT")
    # latest snapshot per side at-or-before decision
    latest: dict[str, QuoteSnapshot] = {}
    for s in usable:
        cur = latest.get(s.side)
        if cur is None or (_utc(s.event_at) > _utc(cur.event_at)):
            latest[s.side] = s
    up = latest.get("UP")
    down = latest.get("DOWN")
    if up is None or down is None:
        return AttachedQuotes(
            up.bid if up else None, up.ask if up else None,
            down.bid if down else None, down.ask if down else None,
            None, None, "MISSING_SIDE",
        )
    if up.event_at is None:
        return AttachedQuotes(None, None, None, None, None, None, "UNKNOWN_TIME")
    age = (dec - _utc(up.event_at)).total_seconds()
    up_mid = (up.bid + up.ask) / 2.0
    # NOTE: down_mid is intentionally NOT derived as 1-up_mid; keep observed.
    return AttachedQuotes(up.bid, up.ask, down.bid, down.ask, up_mid, age, "OK")
