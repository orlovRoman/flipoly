"""Steps 32-33, 36: shared trading-economics function.

Inputs: side, budget ($1 fee-inclusive), available ask levels, size limit,
applicable fee, outcome. Outputs: shares, cost, fee, remainder, PnL.
Checks: win/loss/partial fills reconcile; spread is NOT re-charged on an
ask entry; Polygon gas is separate.

Confirmed vs scenarios: unknown fee -> net_pnl=None (confirmed) plus
separately labelled scenarios; missing depth -> BLOCKED_DATA (never 0 PnL,
never an assumed fill). UNKNOWN never becomes zero fee.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    shares: float
    cost: float
    fee: float | None
    remainder: float
    payout: float
    net_pnl: float | None
    status: str  # FILLED | PARTIAL | BLOCKED_DATA | UNKNOWN_FEE
    detail: str


def execute(budget: float, ask: float | None, fee_rate: float | None,
            outcome_win: bool | None, size_limit: float | None = None,
            gas: float = 0.0) -> Fill:
    """Buy at ASK (entry via ask). Payout is $1/share on win, $0 on loss."""
    if ask is None or ask <= 0 or ask >= 1:
        return Fill(0, 0, None, budget, 0, None, "BLOCKED_DATA", "no depth/ask")
    if outcome_win is None:
        return Fill(0, 0, None, budget, 0, None, "BLOCKED_DATA", "unknown outcome")
    affordable = budget / ask
    shares = affordable if size_limit is None else min(affordable, size_limit)
    # partial fill when depth limit binds
    partial = size_limit is not None and affordable > size_limit
    cost = shares * ask
    remainder = budget - cost
    if fee_rate is None:
        # confirmed result unknown; scenarios must be computed separately
        payout = shares * (1.0 if outcome_win else 0.0)
        return Fill(shares, cost, None, remainder, payout, None,
                    "UNKNOWN_FEE", "fee unknown -> net_pnl=null")
    fee = cost * fee_rate
    # fee comes out of the $1 budget (fee-inclusive): total spend cost+fee<=budget
    if cost + fee > budget + 1e-9:
        # scale down to fit budget incl. fee
        shares = budget / (ask * (1.0 + fee_rate))
        cost = shares * ask
        fee = cost * fee_rate
        remainder = budget - cost - fee
        partial = True
    else:
        remainder = budget - cost - fee
    payout = shares * (1.0 if outcome_win else 0.0)
    net = payout + remainder - budget - gas
    return Fill(shares, cost, fee, remainder, payout, net,
                "PARTIAL" if partial else "FILLED", "ok")
