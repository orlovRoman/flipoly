"""Proxy TWAP math and oracle gap diagnostics in integer E18 units."""

from __future__ import annotations

from math import ceil
from typing import Iterable, Literal

from polyflip.collector.rtds_collector import (
    E18,
    RTDSError,
    parse_price_e18,
    round_half_up_div,
)

__all__ = [
    "E18",
    "InsufficientCoverageError",
    "OfficialOutcome",
    "ProxyDirection",
    "classify_vs_strike",
    "flip_error",
    "gap_abs_bps",
    "gap_bps",
    "moneyness_bps",
    "parse_price_e18",
    "proxy_direction",
    "strike_error",
    "summarize_gap_bps",
    "twap_e18",
]


class InsufficientCoverageError(RTDSError):
    """Raised when a TWAP window has no usable starting observation."""


OfficialOutcome = Literal["UP", "DOWN"]
ProxyDirection = Literal["UP", "DOWN", "TIE"]


def _require_e18(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be integer E18 units")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def twap_e18(points: Iterable[tuple[int, int]], start_ms: int, end_ms: int) -> int:
    """Stepwise time-weighted average over [start_ms, end_ms).

    Points are (timestamp_ms, price_e18) oracle-state observations. The value is
    held constant from the latest observation at or before each instant. A
    window without an observation at or before its start is missing, not zero.
    """
    if isinstance(start_ms, bool) or not isinstance(start_ms, int):
        raise TypeError("start_ms must be int")
    if isinstance(end_ms, bool) or not isinstance(end_ms, int):
        raise TypeError("end_ms must be int")
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("window must satisfy 0 <= start_ms < end_ms")
    ordered = sorted((int(ts), int(price)) for ts, price in points)
    for ts, price in ordered:
        if ts < 0:
            raise ValueError("point timestamp must be non-negative")
        _require_e18("point price", price)
    start_idx: int | None = None
    for idx, (ts, _) in enumerate(ordered):
        if ts <= start_ms:
            start_idx = idx
        else:
            break
    if start_idx is None:
        raise InsufficientCoverageError("no observation at or before window start")

    total = 0
    cursor = start_ms
    price = ordered[start_idx][1]
    idx = start_idx + 1
    while idx < len(ordered) and ordered[idx][0] < end_ms:
        ts, next_price = ordered[idx]
        if ts > cursor:
            total += price * (ts - cursor)
            cursor = ts
        price = next_price
        idx += 1
    total += price * (end_ms - cursor)
    return round_half_up_div(total, end_ms - start_ms)


def gap_bps(proxy_e18: int, official_e18: int) -> int:
    """Signed reconstruction gap: 10,000 * (proxy - official) / official."""
    proxy = _require_e18("proxy_e18", proxy_e18)
    official = _require_e18("official_e18", official_e18)
    diff = proxy - official
    if diff == 0:
        return 0
    quotient, remainder = divmod(10_000 * abs(diff), official)
    if 2 * remainder >= official:
        quotient += 1
    return quotient if diff > 0 else -quotient


def gap_abs_bps(proxy_e18: int, official_e18: int) -> int:
    return abs(gap_bps(proxy_e18, official_e18))


def summarize_gap_bps(values: Iterable[int]) -> dict[str, int | float | None]:
    """Median/P90/P95/P99 over integer basis-point gaps with nearest-rank quantiles."""
    vals = sorted(int(v) for v in values)
    if not vals:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    n = len(vals)

    def rank(pct: float) -> int:
        return vals[max(1, ceil(pct * n / 100.0)) - 1]

    if n % 2 == 1:
        median: int | float = vals[n // 2]
    else:
        low, high = vals[n // 2 - 1], vals[n // 2]
        median = low if low == high else (low + high) / 2.0
    return {
        "count": n,
        "min": vals[0],
        "median": median,
        "p90": rank(90.0),
        "p95": rank(95.0),
        "p99": rank(99.0),
        "max": vals[-1],
    }


def proxy_direction(open_e18: int, close_e18: int) -> ProxyDirection:
    _require_e18("open_e18", open_e18)
    _require_e18("close_e18", close_e18)
    if close_e18 > open_e18:
        return "UP"
    if close_e18 < open_e18:
        return "DOWN"
    return "TIE"


def flip_error(
    proxy_open_e18: int, proxy_close_e18: int, official_up: bool
) -> dict[str, ProxyDirection | bool | None]:
    """Compare proxy move direction with the real official outcome.

    A tied proxy has no direction and therefore no flip error. Proxy direction
    is never presented as the official oracle result.
    """
    direction = proxy_direction(proxy_open_e18, proxy_close_e18)
    if direction == "TIE":
        return {"proxy_direction": direction, "flip_error": None}
    return {
        "proxy_direction": direction,
        "flip_error": (direction == "UP") != bool(official_up),
    }


def classify_vs_strike(
    close_e18: int, strike_e18: int, up_on_equal: bool | None
) -> bool | None:
    """Classify close against a common strike without inventing DRAW."""
    _require_e18("close_e18", close_e18)
    _require_e18("strike_e18", strike_e18)
    if close_e18 > strike_e18:
        return True
    if close_e18 < strike_e18:
        return False
    return None if up_on_equal is None else bool(up_on_equal)


def strike_error(
    proxy_close_e18: int,
    official_strike_e18: int,
    official_up: bool,
    up_on_equal: bool | None,
) -> dict[str, bool | None]:
    """Proxy-versus-official-strike error under one common strike."""
    classified = classify_vs_strike(proxy_close_e18, official_strike_e18, up_on_equal)
    if classified is None:
        return {"classified_up": None, "strike_error": None}
    return {
        "classified_up": classified,
        "strike_error": classified != bool(official_up),
    }


def moneyness_bps(reference_e18: int, strike_e18: int) -> int:
    """Position versus strike in bps. This is not a reconstruction gap."""
    return gap_bps(reference_e18, strike_e18)
