"""Step 14: past volatility from completed, available 1-minute bars.

Fixed spec: 1-minute log-returns over the preceding 30 minutes, using only
bars with close_time <= decision_at. Gaps are never filled with future
values; short histories yield fewer returns (status SHORT_HISTORY).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Bar:
    open_time: datetime
    close_time: datetime
    close: float


@dataclass(frozen=True)
class VolResult:
    sigma_per_sec: float | None
    n_returns: int
    status: str  # OK | SHORT_HISTORY | NO_BARS


BAR_SEC = 60.0
WINDOW_BARS = 30


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def past_volatility(bars: list[Bar], decision_at: datetime,
                    window_bars: int = WINDOW_BARS) -> VolResult:
    dec = _utc(decision_at)
    past = sorted(
        (b for b in bars if _utc(b.close_time) <= dec),
        key=lambda b: _utc(b.close_time),
    )
    closes = [b.close for b in past[-(window_bars + 1):]]
    if len(closes) < 2:
        return VolResult(None, 0, "NO_BARS")
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)
            if closes[i] > 0 and closes[i + 1] > 0]
    if not rets:
        return VolResult(None, 0, "NO_BARS")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    # returns are per-minute; convert stdev to per-second
    sigma_per_sec = math.sqrt(max(var, 0.0)) / math.sqrt(BAR_SEC)
    status = "OK" if len(rets) >= window_bars - 1 else "SHORT_HISTORY"
    return VolResult(sigma_per_sec, len(rets), status)
