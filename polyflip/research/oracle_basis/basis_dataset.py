"""Oracle-basis dataset assembly: horizons, labels, splits, and slices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from polyflip.collector.rtds_collector import RTDSError, parse_timestamp_ms
from polyflip.research.oracle_basis.proxy_twap import (
    flip_error,
    gap_abs_bps,
    gap_bps,
    moneyness_bps,
    strike_error,
)

HORIZONS_SEC = (180, 60, 30, 15, 5)


def horizon_checkpoints(
    end_ms: int, horizons_sec: Iterable[int] = HORIZONS_SEC
) -> dict[int, int]:
    """Decision checkpoints measured backwards from market end."""
    end = parse_timestamp_ms(end_ms)
    horizons = tuple(int(h) for h in horizons_sec)
    if (
        not horizons
        or any(h <= 0 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise RTDSError("horizons must be unique positive seconds")
    checkpoints = {}
    for horizon in sorted(horizons, reverse=True):
        checkpoint = end - horizon * 1000
        if checkpoint < 0:
            raise RTDSError("horizon checkpoint precedes epoch")
        checkpoints[horizon] = checkpoint
    return checkpoints


def assemble_row(
    *,
    market_id: str,
    asset: str,
    horizon_sec: int,
    checkpoint_ms: int,
    spot_status: str,
    spot_age_ms: int | None = None,
    proxy_open_e18: int | None = None,
    proxy_close_e18: int | None = None,
    official_twap_e18: int | None = None,
    official_status: str = "MISSING",
    strike_e18: int | None = None,
    up_on_equal: bool | None = None,
    book_mid_up: float | None = None,
    ask_up: float | None = None,
    ask_down: float | None = None,
    outcome_up: bool | None = None,
    exclusion_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble one horizon row without promoting proxy data to oracle truth."""
    if not str(market_id):
        raise RTDSError("market_id is required")
    if spot_status not in ("OK", "STALE", "NO_DATA"):
        raise RTDSError(f"invalid spot_status: {spot_status!r}")
    if outcome_up is not None and not isinstance(outcome_up, bool):
        raise RTDSError("outcome_up must be True/False/None")

    gap_value: int | None = None
    gap_abs_value: int | None = None
    reconstruction_available = False
    gap_reason = (
        official_status
        if official_twap_e18 is None
        else ("PROXY_TWAP_MISSING" if proxy_close_e18 is None else "OK")
    )
    if official_twap_e18 is not None and proxy_close_e18 is not None:
        gap_value = gap_bps(proxy_close_e18, official_twap_e18)
        gap_abs_value = gap_abs_bps(proxy_close_e18, official_twap_e18)
        reconstruction_available = True

    flip = None
    if (
        proxy_open_e18 is not None
        and proxy_close_e18 is not None
        and outcome_up is not None
    ):
        flip = flip_error(proxy_open_e18, proxy_close_e18, outcome_up)
    strike = None
    if (
        proxy_close_e18 is not None
        and strike_e18 is not None
        and outcome_up is not None
    ):
        strike = strike_error(proxy_close_e18, strike_e18, outcome_up, up_on_equal)
    moneyness = None
    if proxy_close_e18 is not None and strike_e18 is not None:
        moneyness = moneyness_bps(proxy_close_e18, strike_e18)

    return {
        "market_id": str(market_id),
        "asset": str(asset).strip().upper(),
        "horizon_sec": int(horizon_sec),
        "checkpoint_ms": parse_timestamp_ms(checkpoint_ms),
        "spot_status": spot_status,
        "spot_age_ms": spot_age_ms,
        "proxy_open_e18": proxy_open_e18,
        "proxy_close_e18": proxy_close_e18,
        "official_twap_e18": official_twap_e18,
        "official_status": official_status,
        "reconstruction_gap_bps": gap_value,
        "reconstruction_gap_abs_bps": gap_abs_value,
        "reconstruction_available": reconstruction_available,
        "reconstruction_gap_reason": gap_reason,
        "strike_e18": strike_e18,
        "moneyness_bps": moneyness,
        "book_mid_up": book_mid_up,
        "ask_up": ask_up,
        "ask_down": ask_down,
        "outcome_up": outcome_up,
        "label_source": "OFFICIAL_OUTCOME" if outcome_up is not None else "UNRESOLVED",
        "proxy_is_oracle": False,
        "flip_error": flip,
        "strike_error": strike,
        "exclusion_reasons": list(exclusion_reasons),
    }


@dataclass(frozen=True)
class SplitMarket:
    market_id: str
    utc_day: str
    end_ms: int
    start_ms: int = 0


def _utc_midnight_ms(day: str) -> int:
    try:
        year, month, dom = (int(part) for part in str(day).split("-"))
        dt = datetime(year, month, dom, tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise RTDSError(f"invalid utc_day: {day!r}") from exc
    return int(dt.timestamp() * 1000)


def split_markets(
    markets: Iterable[SplitMarket | dict[str, Any]],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    embargo_ms: int = 900_000,
) -> dict[str, Any]:
    """Chronological whole-UTC-day split with a 15-minute embargo window.

    Protocol v0.2: every asset sharing a UTC day stays in one split. For each
    split boundary (UTC midnight), drop (a) any market whose [start_ms, end_ms]
    crosses the boundary (feature/label intervals must not touch the next
    split), and (b) earlier-split markets ending inside `embargo_ms` before the
    boundary. Markets without a known start_ms cannot prove non-overlap and are
    excluded as NO_START_MS.
    """
    normalized: list[SplitMarket] = []
    excluded: list[dict[str, str]] = []
    for item in markets:
        if isinstance(item, SplitMarket):
            market = item
        else:
            market = SplitMarket(
                market_id=str(item["market_id"]),
                utc_day=str(item["utc_day"]),
                end_ms=parse_timestamp_ms(item["end_ms"]),
                start_ms=(
                    parse_timestamp_ms(item["start_ms"])
                    if item.get("start_ms") is not None
                    else 0
                ),
            )
        if market.start_ms and market.start_ms >= market.end_ms:
            raise RTDSError(f"market {market.market_id}: start_ms must precede end_ms")
        if not market.start_ms:
            excluded.append({"market_id": market.market_id, "reason": "NO_START_MS"})
            continue
        normalized.append(market)
    if len({m.market_id for m in normalized}) != len(normalized):
        raise RTDSError("split input must contain unique markets")
    if (
        isinstance(embargo_ms, bool)
        or not isinstance(embargo_ms, int)
        or embargo_ms < 0
    ):
        raise RTDSError("embargo_ms must be a non-negative int")
    days = sorted({m.utc_day for m in normalized})
    if len(days) < 3:
        raise RTDSError("need at least three UTC days for train/validation/test")
    total = train_frac + val_frac + test_frac
    if total <= 0:
        raise RTDSError("split fractions must be positive")
    train_n = max(1, min(len(days) - 2, int(len(days) * train_frac / total)))
    val_n = max(1, min(len(days) - train_n - 1, int(len(days) * val_frac / total)))
    test_n = len(days) - train_n - val_n
    if test_n < 1:
        raise RTDSError("split fractions leave no test day")
    day_split = {}
    day_split.update({d: "train" for d in days[:train_n]})
    day_split.update({d: "validation" for d in days[train_n : train_n + val_n]})
    day_split.update({d: "test" for d in days[train_n + val_n :]})

    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    by_id = {m.market_id: m for m in normalized}
    for market in sorted(normalized, key=lambda m: (m.end_ms, m.market_id)):
        splits[day_split[market.utc_day]].append(market.market_id)
    dropped: list[dict[str, str]] = []
    for earlier_name, later_name in (("train", "validation"), ("validation", "test")):
        earlier_days = sorted(d for d, s in day_split.items() if s == earlier_name)
        later_days = sorted(d for d, s in day_split.items() if s == later_name)
        if not earlier_days or not later_days:
            continue
        boundary_ms = _utc_midnight_ms(later_days[0])
        for split_name in (earlier_name, later_name):
            for mid in list(splits[split_name]):
                m = by_id[mid]
                if m.start_ms < boundary_ms < m.end_ms:
                    splits[split_name].remove(mid)
                    dropped.append(
                        {"market_id": mid, "reason": "CROSSES_SPLIT_BOUNDARY"}
                    )
        for mid in list(splits[earlier_name]):
            m = by_id[mid]
            if m.end_ms > boundary_ms - embargo_ms:
                splits[earlier_name].remove(mid)
                dropped.append(
                    {
                        "market_id": mid,
                        "reason": f"EMBARGO_WINDOW_BEFORE_{later_name.upper()}",
                    }
                )
    return {
        "days": day_split,
        "splits": splits,
        "dropped": dropped,
        "excluded": excluded,
        "fractions": {"train": train_frac, "validation": val_frac, "test": test_frac},
        "embargo_ms": embargo_ms,
    }


def gap_bucket(abs_gap_bps: int | None) -> str:
    if abs_gap_bps is None:
        return "NO_OFFICIAL_REFERENCE"
    if (
        isinstance(abs_gap_bps, bool)
        or not isinstance(abs_gap_bps, int)
        or abs_gap_bps < 0
    ):
        raise RTDSError("abs_gap_bps must be a non-negative int or None")
    if abs_gap_bps < 3:
        return "<3bps"
    if abs_gap_bps <= 10:
        return "3-10bps"
    return ">10bps"


def book_regime(mid_up: float) -> str:
    if (
        isinstance(mid_up, bool)
        or not isinstance(mid_up, (float, int))
        or not 0.0 <= float(mid_up) <= 1.0
    ):
        raise RTDSError("mid_up must be a probability in [0, 1]")
    # Integer basis points keep the |mid - 0.50| < 0.10 boundary exact despite
    # binary-float representation (float(0.40) - 0.50 is 0.0999... otherwise).
    bps = int(round(float(mid_up) * 10_000))
    return "CONTESTED" if abs(bps - 5_000) < 1_000 else "FAVORITE"
