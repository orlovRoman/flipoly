"""Steps 32-33, 36: shared trading-economics function.

Single source of truth for fills and fees: the vendored
``polyflip.research.orderbook_execution`` (branch ``feature/paper-ct-outsider``,
commit ``2b531a2``). No second independent fee calculator lives here — this
module only wraps it with the study's fee-unknown policy and scenarios.

Budget semantics (follow the vendored module + BTC_CT_T5_V1):
- an order commits up to $1 of PURCHASE COST (``max_purchase_cost_usdc``);
- taker fee is accounted SEPARATELY on top (``spent * fee_rate``);
- unspent budget is NEVER booked as loss;
- spread is NOT re-charged on an ask entry (entry walks the ask ladder);
- Polygon gas is accounted separately.

Fee policy (0.002 is NOT confirmed as the actual commission):
- ``fee_rate=None`` -> ``fee_status="UNKNOWN"``, ``actual_fee=None``,
  ``net_pnl=None`` (confirmed). Gross PnL is still reported.
- separately labelled SCENARIOS at 0 / 0.001 / 0.002 of purchase cost
  (``scenario_pnls``) — signed as scenarios, never as a realistic range.
- ``breakeven_fee_rate`` + ``sensitivity`` for the threshold analysis.
- absence of a fee in an API answer never means zero fee; the current
  scheme never auto-applies to history.

Depth policy: no ask/depth -> ``BLOCKED_DATA`` (never 0 PnL, never an
assumed fill). Top-of-book-only fills are marked ``TOP_ONLY_ASSUMED``;
a truncated ladder that runs out maps to ``DEPTH_EXHAUSTED_UNKNOWN``.
"""
from __future__ import annotations

from dataclasses import dataclass

from polyflip.research.orderbook_execution import (
    calculate_trade_payout_and_pnl,
    simulate_orderbook_execution,
)

SCENARIO_RATES = (0.0, 0.001, 0.002)


@dataclass(frozen=True)
class Fill:
    shares: float
    cost: float            # spent purchase cost (<= budget)
    fee: float | None      # None when fee UNKNOWN
    remainder: float       # unspent budget (never booked as loss)
    payout: float
    gross_pnl: float | None
    net_pnl: float | None  # None when fee UNKNOWN (confirmed)
    status: str            # FILLED | PARTIAL | BLOCKED_DATA | UNKNOWN_FEE
    detail: str
    fee_status: str = "UNKNOWN"   # CONFIRMED | UNKNOWN
    data_status: str = "CONFIRMED"


def _blocked(budget: float, detail: str) -> Fill:
    return Fill(0.0, 0.0, None, budget, 0.0, None, None,
                "BLOCKED_DATA", detail, "UNKNOWN", "CONFIRMED")


def execute(budget: float, ask: float | None, fee_rate: float | None,
            outcome_win: bool | None, size_limit: float | None = None,
            gas: float = 0.0, ask_levels: list | None = None,
            is_truncated: bool = False,
            book_age_sec: float | None = 0.0) -> Fill:
    """Buy at ASK levels. Payout is $1/share on win, $0 on loss."""
    if ask is None or not (0.0 < ask < 1.0):
        return _blocked(budget, "no depth/ask")
    if outcome_win is None:
        return _blocked(budget, "unknown outcome")
    if ask_levels is None:
        # Top-of-book only, no size info: fill exactly the affordable size at
        # the observed ask. Depth beyond the top is ASSUMED, not confirmed.
        depth_assumed = True
        ladder = [(ask, budget / ask)]
    else:
        depth_assumed = False
        ladder = list(ask_levels)
    if size_limit is not None:
        # Depth-imposed cap: clip the ladder to the size limit.
        clipped: list[tuple[float, float]] = []
        left = size_limit
        for p, s in ladder:
            take = min(s, left)
            if take > 0:
                clipped.append((p, take))
                left -= take
            if left <= 0:
                break
        ladder = clipped

    fill = simulate_orderbook_execution(
        asks=[{"price": p, "size": s} for p, s in ladder],
        budget_usdc=budget,
        price_limit=ask,
        taker_fee_rate=fee_rate if fee_rate is not None else 0.0,
        is_truncated=is_truncated or depth_assumed,
        book_age_sec=book_age_sec if book_age_sec is not None else 0.0,
    )
    if fill.fill_status == "STALE_BOOK":
        return _blocked(budget, "stale book")
    if fill.fill_status in ("UNFILLED", "LIMIT_EXCEEDED"):
        return _blocked(budget, f"unfilled: {fill.fill_status}")

    data_status = fill.data_status
    if depth_assumed and data_status == "CONFIRMED":
        data_status = "TOP_ONLY_ASSUMED"
    status = "FILLED" if fill.fill_status == "FULL" else "PARTIAL"
    if fill.data_status == "TRUNCATED_UNKNOWN":
        status = "PARTIAL"
        data_status = "DEPTH_EXHAUSTED_UNKNOWN"
    if size_limit is not None and fill.filled_shares < budget / ask - 1e-9:
        status = "PARTIAL"

    target = 1 if outcome_win else 0
    if fee_rate is None:
        gross = fill.filled_shares * target - fill.spent_usdc
        return Fill(fill.filled_shares, fill.spent_usdc, None,
                    fill.remaining_budget, fill.filled_shares * target,
                    gross, None, "UNKNOWN_FEE",
                    "fee unknown -> net_pnl=null; see scenario_pnls",
                    "UNKNOWN", data_status)
    st = calculate_trade_payout_and_pnl(
        filled_shares=fill.filled_shares, spent_usdc=fill.spent_usdc,
        budget_usdc=budget, target=target, fee=fill.fee)
    net = st.net_pnl - gas
    return Fill(fill.filled_shares, fill.spent_usdc, fill.fee,
                fill.remaining_budget, st.payout, st.gross_pnl, net,
                status, "ok", "CONFIRMED", data_status)


def scenario_pnls(budget: float, ask: float | None, outcome_win: bool | None,
                  rates: tuple = SCENARIO_RATES,
                  **kw) -> dict[float, float | None]:
    """Signed fee scenarios (never a confirmed result)."""
    return {r: execute(budget, ask, r, outcome_win, **kw).net_pnl for r in rates}


def breakeven_fee_rate(p_win: float, ask: float) -> float | None:
    """Fee rate where E[net]==0 for a FULL $1 fill: r* = p/ask - 1."""
    if not (0.0 < ask < 1.0) or not (0.0 <= p_win <= 1.0):
        return None
    return p_win / ask - 1.0


def sensitivity(budget: float, ask: float | None, p_win: float,
                rates: tuple = SCENARIO_RATES) -> dict[float, float | None]:
    """Expected net across fee scenarios for a probabilistic edge."""
    out = {}
    for r in rates:
        f_win = execute(budget, ask, r, True)
        f_loss = execute(budget, ask, r, False)
        if f_win.net_pnl is None or f_loss.net_pnl is None:
            out[r] = None
        else:
            out[r] = p_win * f_win.net_pnl + (1 - p_win) * f_loss.net_pnl
    return out
