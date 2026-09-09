"""
polyflip/collector/orderbook_depth.py

Orderbook contract, quality validation, and level normalization (Points 4, 6, 7, 8).
Ensures:
- Real orderbooks for both YES and NO tokens without surrogate reconstructions.
- Unambiguous units: size in shares (акции), price * size in USDC.
- Distinction between missing liquidity (None) and zero.
- Rigorous quality checks: ordering, negative/non-finite sizes, duplicates, crossed books, sequence continuity.
- Orderbook completeness metrics and coverage tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence
import math


@dataclass(frozen=True)
class OrderbookLevel:
    price: float
    size: float  # Shares (акции)

    @property
    def usdc_depth(self) -> float:
        """Price * size in USDC."""
        return self.price * self.size


@dataclass
class OrderbookContract:
    market_id: str
    token_id: str
    outcome_side: str  # "YES" or "NO"
    event_at: datetime
    received_at: datetime
    bids: list[dict[str, float]]
    asks: list[dict[str, float]]
    sequence_id: str | None = None
    is_truncated: bool = False
    depth_limit: int | None = None
    source: str = "CLOB"
    quality_status: str = "VALID"
    quality_notes: str | None = None

    # Precalculated derived metrics
    best_bid_price: float | None = None
    best_bid_size: float | None = None  # Shares
    best_ask_price: float | None = None
    best_ask_size: float | None = None  # Shares
    depth_usdc_bid: float | None = None  # Total bids USDC
    depth_usdc_ask: float | None = None  # Total asks USDC


def validate_and_normalize_orderbook(
    raw_bids: Sequence[dict[str, Any]],
    raw_asks: Sequence[dict[str, Any]],
    market_id: str,
    token_id: str,
    outcome_side: str,
    event_at: datetime | None = None,
    received_at: datetime | None = None,
    sequence_id: str | int | None = None,
    depth_limit: int | None = None,
    source: str = "CLOB",
    expected_sequence_id: str | int | None = None,
) -> OrderbookContract:
    """
    Validates and normalizes orderbook depth ladders according to Points 4 & 7.

    Self-checks:
    - Size is in shares, price * size in USDC.
    - Missing book side returns None for best price/size (not 0.0).
    - Checks level ordering, negative sizes, duplicate prices, crossed books, and sequence gaps.
    """
    now = datetime.now(timezone.utc)
    ev_dt = event_at or now
    rec_dt = received_at or now

    side_clean = str(outcome_side).strip().upper()
    if side_clean in ("UP", "YES"):
        side_clean = "YES"
    elif side_clean in ("DOWN", "NO"):
        side_clean = "NO"

    issues: list[str] = []
    status = "VALID"

    # 1. Parse bids
    norm_bids: list[dict[str, float]] = []
    seen_bid_prices: set[float] = set()
    prev_bid_price = float("inf")
    unordered_bids = False

    for item in raw_bids:
        try:
            p = float(item["price"])
            raw_s = item.get("size") if item.get("size") is not None else item.get("quantity")
            if raw_s is None:
                raise ValueError("Missing size")
            s = float(raw_s)
        except (AttributeError, TypeError, ValueError, KeyError):
            issues.append("INVALID_NUMERIC_FORMAT_IN_BIDS")
            status = "CORRUPTED"
            continue

        if not (math.isfinite(p) and math.isfinite(s)):
            issues.append("NON_FINITE_NUMBERS_IN_BIDS")
            status = "CORRUPTED"
            continue

        if p < 0.0 or s < 0.0:
            issues.append("NEGATIVE_PRICE_OR_SIZE_IN_BIDS")
            status = "NEGATIVE_SIZE"
            continue

        if p in seen_bid_prices:
            issues.append(f"DUPLICATE_BID_PRICE_{p}")
            status = "DUPLICATE_PRICES"
        seen_bid_prices.add(p)

        if p > prev_bid_price:
            unordered_bids = True
        prev_bid_price = p

        norm_bids.append({"price": p, "size": s})

    # 2. Parse asks
    norm_asks: list[dict[str, float]] = []
    seen_ask_prices: set[float] = set()
    prev_ask_price = float("-inf")
    unordered_asks = False

    for item in raw_asks:
        try:
            p = float(item["price"])
            raw_s = item.get("size") if item.get("size") is not None else item.get("quantity")
            if raw_s is None:
                raise ValueError("Missing size")
            s = float(raw_s)
        except (AttributeError, TypeError, ValueError, KeyError):
            issues.append("INVALID_NUMERIC_FORMAT_IN_ASKS")
            status = "CORRUPTED"
            continue

        if not (math.isfinite(p) and math.isfinite(s)):
            issues.append("NON_FINITE_NUMBERS_IN_ASKS")
            status = "CORRUPTED"
            continue

        if p < 0.0 or s < 0.0:
            issues.append("NEGATIVE_PRICE_OR_SIZE_IN_ASKS")
            status = "NEGATIVE_SIZE"
            continue

        if p in seen_ask_prices:
            issues.append(f"DUPLICATE_ASK_PRICE_{p}")
            status = "DUPLICATE_PRICES"
        seen_ask_prices.add(p)

        if p < prev_ask_price:
            unordered_asks = True
        prev_ask_price = p

        norm_asks.append({"price": p, "size": s})

    if unordered_bids or unordered_asks:
        # Bids must be descending, asks ascending
        norm_bids.sort(key=lambda x: x["price"], reverse=True)
        norm_asks.sort(key=lambda x: x["price"], reverse=False)
        if status == "VALID":
            status = "UNORDERED_LEVELS_NORMALIZED"
        issues.append("UNORDERED_LEVELS")

    # Sequence tracking (Point 7)
    seq_str = str(sequence_id) if sequence_id is not None else None
    if expected_sequence_id is not None and seq_str is not None:
        try:
            if int(seq_str) != int(expected_sequence_id):
                issues.append(f"SEQUENCE_GAP: expected {expected_sequence_id}, got {seq_str}")
                if status == "VALID":
                    status = "SEQUENCE_GAP"
        except (ValueError, TypeError):
            pass

    # Top-of-book metrics
    best_bid_price: float | None = None
    best_bid_size: float | None = None
    depth_usdc_bid: float | None = None

    if norm_bids:
        # Highest bid is first after sort
        best_bid_price = norm_bids[0]["price"]
        best_bid_size = norm_bids[0]["size"]
        depth_usdc_bid = round(sum(b["price"] * b["size"] for b in norm_bids), 6)
    else:
        # Missing volume distinct from zero
        best_bid_price = None
        best_bid_size = None
        depth_usdc_bid = None

    best_ask_price: float | None = None
    best_ask_size: float | None = None
    depth_usdc_ask: float | None = None

    if norm_asks:
        # Lowest ask is first after sort
        best_ask_price = norm_asks[0]["price"]
        best_ask_size = norm_asks[0]["size"]
        depth_usdc_ask = round(sum(a["price"] * a["size"] for a in norm_asks), 6)
    else:
        best_ask_price = None
        best_ask_size = None
        depth_usdc_ask = None

    # Empty book check
    if not norm_bids and not norm_asks:
        status = "EMPTY_BOOK"
        issues.append("NO_BIDS_OR_ASKS")
    elif not norm_bids:
        issues.append("EMPTY_BIDS")
    elif not norm_asks:
        issues.append("EMPTY_ASKS")

    # Crossed book check
    if best_bid_price is not None and best_ask_price is not None:
        if best_bid_price >= best_ask_price:
            status = "CROSSED_BOOK"
            issues.append(f"CROSSED_BOOK_BID_{best_bid_price}_GE_ASK_{best_ask_price}")

    is_truncated = False
    if depth_limit is not None and depth_limit > 0:
        if len(norm_bids) >= depth_limit or len(norm_asks) >= depth_limit:
            is_truncated = True

    notes = "; ".join(issues) if issues else None

    return OrderbookContract(
        market_id=str(market_id),
        token_id=str(token_id),
        outcome_side=side_clean,
        event_at=ev_dt,
        received_at=rec_dt,
        bids=norm_bids,
        asks=norm_asks,
        sequence_id=seq_str,
        is_truncated=is_truncated,
        depth_limit=depth_limit,
        source=source,
        quality_status=status,
        quality_notes=notes,
        best_bid_price=best_bid_price,
        best_bid_size=best_bid_size,
        best_ask_price=best_ask_price,
        best_ask_size=best_ask_size,
        depth_usdc_bid=depth_usdc_bid,
        depth_usdc_ask=depth_usdc_ask,
    )


def compute_orderbook_completeness_report(
    contracts: Sequence[OrderbookContract],
    decision_market_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Calculates Point 8 telemetry on recorded orderbooks:
    - fraction of decisions with both sides (YES and NO)
    - age of latest snapshot
    - distribution of intervals between snapshots
    - fraction of truncated and invalid books
    """
    if not contracts:
        return {
            "total_contracts": 0,
            "both_sides_coverage_pct": 0.0,
            "latest_snapshot_age_sec": None,
            "mean_interval_sec": None,
            "median_interval_sec": None,
            "truncated_fraction": 0.0,
            "invalid_fraction": 0.0,
            "status_breakdown": {},
        }

    now = datetime.now(timezone.utc)
    market_time_sides: dict[tuple[str, datetime], set[str]] = {}
    market_sides: dict[str, set[str]] = {}
    timestamps: list[datetime] = []
    n_truncated = 0
    n_invalid = 0
    status_counts: dict[str, int] = {}

    for c in contracts:
        market_time_sides.setdefault((c.market_id, c.received_at), set()).add(c.outcome_side)
        market_sides.setdefault(c.market_id, set()).add(c.outcome_side)
        timestamps.append(c.received_at)
        if c.is_truncated:
            n_truncated += 1
        if c.quality_status not in ("VALID", "UNORDERED_LEVELS_NORMALIZED"):
            n_invalid += 1
        status_counts[c.quality_status] = status_counts.get(c.quality_status, 0) + 1

    timestamps.sort()
    intervals = [
        (timestamps[i] - timestamps[i - 1]).total_seconds()
        for i in range(1, len(timestamps))
    ]

    target_markets = set(decision_market_ids) if decision_market_ids else set(market_sides.keys())
    
    both_count = 0
    total_decision_snaps = 0
    
    for m in target_markets:
        c_list = [c for c in contracts if c.market_id == m]
        c_list.sort(key=lambda x: x.received_at)
        
        # Consider each YES as a target snapshot
        yes_books = [c for c in c_list if c.outcome_side == "YES"]
        no_books = [c for c in c_list if c.outcome_side == "NO"]
        
        for yb in yes_books:
            total_decision_snaps += 1
            # Directed causal search: NO book must be received at or after YES within 5.0 seconds
            # 0.0 <= (nb.received_at - yb.received_at).total_seconds() <= 5.0
            paired = any(
                0.0 <= (nb.received_at - yb.received_at).total_seconds() <= 5.0
                for nb in no_books
            )
            if paired:
                both_count += 1

    both_pct = (both_count / total_decision_snaps * 100.0) if total_decision_snaps > 0 else 0.0

    latest_age = (now - timestamps[-1]).total_seconds() if timestamps else None
    mean_int = float(sum(intervals) / len(intervals)) if intervals else None
    med_int = float(sorted(intervals)[len(intervals) // 2]) if intervals else None

    return {
        "total_contracts": len(contracts),
        "total_markets": len(target_markets),
        "both_sides_coverage_pct": round(both_pct, 2),
        "latest_snapshot_age_sec": round(latest_age, 2) if latest_age is not None else None,
        "mean_interval_sec": round(mean_int, 2) if mean_int is not None else None,
        "median_interval_sec": round(med_int, 2) if med_int is not None else None,
        "truncated_fraction": round(n_truncated / len(contracts), 4),
        "invalid_fraction": round(n_invalid / len(contracts), 4),
        "status_breakdown": status_counts,
    }
