"""
polyflip/research/orderbook_execution.py

Pure functions for orderbook depth execution simulation, fee modeling,
historical latency delays, and volume reuse prevention (Points 14, 15, 16, 17, 26, 27).

Requirements covered:
- Pure execution function over orderbook depth ladders.
- Unambiguous units: size in shares, price * size in USDC.
- Truncated depth: outcomes beyond observed ladder are strictly UNKNOWN.
- Partial fill payout & fee accounting: remaining budget is never booked as loss.
- Arrival latency & staleness limits: stale snapshots return UNCERTAIN / STALE_BOOK.
- Unique decision tracking and volume consumption: multiple orders cannot independently
  consume the same snapshot volume without a confirmed update.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any
import math


@dataclass(frozen=True)
class OrderbookFillResult:
    filled_shares: float
    spent_usdc: float
    remaining_budget: float
    vwap: float | None
    fee: float
    fill_status: str  # "FULL", "PARTIAL", "UNFILLED", "LIMIT_EXCEEDED", "STALE_BOOK", "DEPTH_EXHAUSTED_UNKNOWN"
    data_status: str  # "CONFIRMED", "TRUNCATED_UNKNOWN", "STALE", "UNCERTAIN"
    levels_consumed: int


@dataclass(frozen=True)
class TradeSettlementResult:
    filled_shares: float
    spent_usdc: float
    unspent_budget: float
    payout: float
    gross_pnl: float
    fee: float
    net_pnl: float
    return_on_spent: float
    target: int  # 1 (win) or 0 (loss)


def simulate_orderbook_execution(
    asks: Sequence[dict[str, float] | Sequence[float]],
    budget_usdc: float,
    price_limit: float | None = None,
    taker_fee_rate: float = 0.0,
    is_truncated: bool = False,
    arrival_delay_sec: float = 0.0,
    book_age_sec: float = 0.0,
    max_staleness_sec: float = 15.0,
    already_consumed_shares_by_level: Sequence[float] | None = None,
) -> OrderbookFillResult:
    """
    Pure function executing a market buy order against ask levels (Point 14 & 16).

    Parameters:
    - asks: sorted ascending by price, each level is {"price": p, "size": s}
    - budget_usdc: total capital allocated for this trade (e.g. $1.0, $5.0, $10.0)
    - price_limit: maximum allowable execution price (e.g. 0.40 or 0.03)
    - taker_fee_rate: taker fee fraction (e.g. 0.002 or 0.0)
    - is_truncated: whether the orderbook was capped at fixed depth (e.g. top 10/20)
    - arrival_delay_sec: simulated network/dispatch latency
    - book_age_sec: age of snapshot at decision moment
    - max_staleness_sec: maximum acceptable book age before declaring state uncertain

    Self-check Point 14:
    For levels 10 x 0.03 and 20 x 0.04:
    - budget $1 without fee buys 27.5 shares, VWAP ~ 0.03636.
    - with price_limit 0.03, only 10 shares are bought.
    """
    if budget_usdc <= 0.0:
        return OrderbookFillResult(
            filled_shares=0.0,
            spent_usdc=0.0,
            remaining_budget=0.0,
            vwap=None,
            fee=0.0,
            fill_status="UNFILLED",
            data_status="CONFIRMED",
            levels_consumed=0,
        )

    # Point 16: Latency and staleness safeguards
    if book_age_sec is None:
        return OrderbookFillResult(
            filled_shares=0.0,
            spent_usdc=0.0,
            remaining_budget=budget_usdc,
            vwap=None,
            fee=0.0,
            fill_status="STALE_BOOK",
            data_status="UNKNOWN_TIMESTAMP",
            levels_consumed=0,
        )

    total_latency_sec = book_age_sec + arrival_delay_sec
    if total_latency_sec > max_staleness_sec:
        return OrderbookFillResult(
            filled_shares=0.0,
            spent_usdc=0.0,
            remaining_budget=budget_usdc,
            vwap=None,
            fee=0.0,
            fill_status="STALE_BOOK",
            data_status="STALE",
            levels_consumed=0,
        )

    # Normalize ask levels
    parsed_asks: list[tuple[float, float]] = []
    for item in asks:
        if isinstance(item, dict):
            p = float(item.get("price", 0.0))
            s = float(item.get("size", item.get("quantity", 0.0)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            p = float(item[0])
            s = float(item[1])
        else:
            continue
        if p > 0.0 and s > 0.0 and math.isfinite(p) and math.isfinite(s):
            parsed_asks.append((p, s))

    # Sort ascending by price
    parsed_asks.sort(key=lambda x: x[0])

    if not parsed_asks:
        return OrderbookFillResult(
            filled_shares=0.0,
            spent_usdc=0.0,
            remaining_budget=budget_usdc,
            vwap=None,
            fee=0.0,
            fill_status="UNFILLED",
            data_status="CONFIRMED",
            levels_consumed=0,
        )

    remaining_budget = float(budget_usdc)
    total_shares = 0.0
    total_spent = 0.0
    levels_consumed = 0

    for idx, (p, available_size) in enumerate(parsed_asks):
        # Adjust for prior volume consumption on same snapshot (Point 17)
        if already_consumed_shares_by_level is not None and idx < len(already_consumed_shares_by_level):
            available_size = max(0.0, available_size - already_consumed_shares_by_level[idx])

        if available_size <= 1e-9:
            continue

        # Respect limit price
        if price_limit is not None and p > (price_limit + 1e-9):
            break

        levels_consumed += 1
        level_cost = p * available_size

        if remaining_budget >= level_cost:
            # Consume full level
            total_shares += available_size
            total_spent += level_cost
            remaining_budget -= level_cost
        else:
            # Consume partial level up to remaining budget
            partial_shares = remaining_budget / p
            total_shares += partial_shares
            total_spent += remaining_budget
            remaining_budget = 0.0
            break

    vwap = (total_spent / total_shares) if total_shares > 1e-9 else None
    fee = total_spent * taker_fee_rate

    # Determine statuses
    if remaining_budget <= 1e-7:
        fill_status = "FULL"
        data_status = "CONFIRMED"
    elif total_shares > 1e-9:
        if is_truncated and levels_consumed == len(parsed_asks):
            # Depth ran out on truncated book: outcome beyond observed depth is strictly unknown (Point 14)
            fill_status = "DEPTH_EXHAUSTED_UNKNOWN"
            data_status = "TRUNCATED_UNKNOWN"
        else:
            fill_status = "PARTIAL"
            data_status = "CONFIRMED"
    else:
        if price_limit is not None and parsed_asks[0][0] > price_limit:
            fill_status = "LIMIT_EXCEEDED"
        else:
            fill_status = "UNFILLED"
        data_status = "CONFIRMED"

    return OrderbookFillResult(
        filled_shares=round(total_shares, 6),
        spent_usdc=round(total_spent, 6),
        remaining_budget=round(remaining_budget, 6),
        vwap=round(vwap, 6) if vwap is not None else None,
        fee=round(fee, 6),
        fill_status=fill_status,
        data_status=data_status,
        levels_consumed=levels_consumed,
    )


def calculate_trade_payout_and_pnl(
    filled_shares: float,
    spent_usdc: float,
    budget_usdc: float,
    target: int,
    fee: float,
) -> TradeSettlementResult:
    """
    Computes trade outcome and PnL strictly from actually bought shares (Point 15).

    Self-checks:
    - 10 shares for 0.30 USDC:
      * Win (target=1): +9.70 before fee (10.0 - 0.30)
      * Loss (target=0): -0.30 before fee (0.0 - 0.30)
    - Unspent budget is NOT booked as loss.
    - Fee is deducted once.
    """
    unspent_budget = max(0.0, budget_usdc - spent_usdc)
    payout = float(filled_shares) * float(target)
    gross_pnl = payout - spent_usdc
    net_pnl = gross_pnl - fee
    return_on_spent = (net_pnl / spent_usdc) if spent_usdc > 1e-9 else 0.0

    return TradeSettlementResult(
        filled_shares=round(filled_shares, 6),
        spent_usdc=round(spent_usdc, 6),
        unspent_budget=round(unspent_budget, 6),
        payout=round(payout, 6),
        gross_pnl=round(gross_pnl, 6),
        fee=round(fee, 6),
        net_pnl=round(net_pnl, 6),
        return_on_spent=round(return_on_spent, 6),
        target=int(target),
    )


class ExecutionVolumeTracker:
    """
    Tracks executed decisions and consumed orderbook volume (Point 17).
    Guarantees:
    - Unique decision ID prevents double-processing of identical trading decisions.
    - Total consumed volume across multiple orders cannot exceed snapshot liquidity
      without confirmed update.
    """

    def __init__(self):
        self._processed_decision_ids: set[str] = set()
        self._consumed_shares_by_snapshot: dict[str, list[float]] = {}

    def make_decision_id(self, market_id: str, decision_at: Any, outcome_side: str) -> str:
        return f"{market_id}_{pd_ts(decision_at)}_{outcome_side.upper()}"

    def has_decision(self, decision_id: str) -> bool:
        return decision_id in self._processed_decision_ids

    def register_decision(self, decision_id: str) -> bool:
        if decision_id in self._processed_decision_ids:
            return False
        self._processed_decision_ids.add(decision_id)
        return True

    def get_consumed_shares(self, snapshot_key: str, n_levels: int) -> list[float]:
        if snapshot_key not in self._consumed_shares_by_snapshot:
            self._consumed_shares_by_snapshot[snapshot_key] = [0.0] * n_levels
        return self._consumed_shares_by_snapshot[snapshot_key]

    def record_fill(self, snapshot_key: str, asks: Sequence[dict[str, float]], filled_shares: float) -> None:
        consumed = self.get_consumed_shares(snapshot_key, len(asks))
        rem = filled_shares
        for i, level in enumerate(asks):
            if rem <= 1e-9:
                break
            p = float(level.get("price", 0.0))
            s = float(level.get("size", 0.0))
            avail = max(0.0, s - consumed[i])
            take = min(avail, rem)
            consumed[i] += take
            rem -= take


def pd_ts(dt: Any) -> str:
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)
