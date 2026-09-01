"""Conservative uncertainty-aware position sizing for weighted policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from math import isfinite

from polyflip.trading.weighted_policy import clamp_probability


@dataclass(frozen=True)
class SizingDecision:
    p_estimate: float
    p_lower: float
    edge_lower: float
    kelly_fraction: float
    size_multiplier: float
    reason: str


def probability_lower_bound(
    p_estimate: float,
    standard_error: Optional[float],
    *,
    z_score: float = 1.96,
) -> float:
    """One-sided normal lower bound, clipped to a valid probability."""
    p = clamp_probability(p_estimate, 0.5)
    uncertainty = max(0.0, float(standard_error or 0.0))
    z = max(0.0, float(z_score))
    assert p is not None
    return max(0.0, min(1.0, p - z * uncertainty))


def fractional_kelly_fraction(
    p_win: float,
    price: float,
    cost_per_share: float = 0.0,
    *,
    fraction: float = 0.25,
) -> float:
    """Return fractional Kelly for a binary share, after per-share costs."""
    p = clamp_probability(p_win, 0.5)
    q = 1.0 - p if p is not None else 0.5
    price_plus_cost = max(1e-9, float(price) + max(0.0, float(cost_per_share)))
    win_profit = max(1e-9, 1.0 - price_plus_cost)
    odds = win_profit / price_plus_cost
    raw = (odds * p - q) / odds
    return max(0.0, min(1.0, raw * max(0.0, float(fraction))))


DEFAULT_STEPPED_EDGE_THRESHOLDS: tuple[float, float, float] = (0.03, 0.06, 0.10)


def stepped_bet_size(
    edge_lower: float,
    *,
    base_bet_usdc: float = 1.0,
    cap_usdc: float = 3.0,
    edge_thresholds: tuple[float, float, float] = DEFAULT_STEPPED_EDGE_THRESHOLDS,
) -> float:
    """Return a conservative $1 -> $1.5 -> $2 -> $3 stake by lower-bound edge.

    The first level is deliberately the fallback for missing or weak evidence.
    Thresholds are net USDC edge per share and comparisons are inclusive.
    """
    try:
        base = float(base_bet_usdc)
    except (TypeError, ValueError, OverflowError):
        base = 1.0
    try:
        cap = float(cap_usdc)
    except (TypeError, ValueError, OverflowError):
        cap = 3.0
    if not isfinite(base) or base < 0.0:
        base = 1.0
    if not isfinite(cap) or cap < 0.0:
        cap = 3.0
    if cap <= 0.0:
        return 0.0
    if base <= 0.0:
        base = min(1.0, cap)
    try:
        thresholds = tuple(float(value) for value in edge_thresholds)
    except (TypeError, ValueError, OverflowError):
        thresholds = DEFAULT_STEPPED_EDGE_THRESHOLDS
    if len(thresholds) != 3 or any(
        not isfinite(value) for value in thresholds
    ) or tuple(sorted(thresholds)) != thresholds:
        thresholds = DEFAULT_STEPPED_EDGE_THRESHOLDS
    try:
        edge = float(edge_lower)
    except (TypeError, ValueError, OverflowError):
        edge = float("-inf")
    levels = (base, base * 1.5, base * 2.0, base * 3.0)
    if not isfinite(edge):
        selected = levels[0]
    elif edge >= thresholds[2]:
        selected = levels[3]
    elif edge >= thresholds[1]:
        selected = levels[2]
    elif edge >= thresholds[0]:
        selected = levels[1]
    else:
        selected = levels[0]
    return round(max(0.0, min(cap, selected)), 8)



def conservative_size(
    p_estimate: float,
    *,
    price: float,
    cost_per_share: float,
    standard_error: Optional[float],
    fraction: float = 0.25,
    min_edge_lower: float = 0.0,
) -> SizingDecision:
    p = clamp_probability(p_estimate, 0.5)
    assert p is not None
    p_lower = probability_lower_bound(p, standard_error)
    edge_lower = p_lower - float(price) - max(0.0, float(cost_per_share))
    if edge_lower < float(min_edge_lower):
        return SizingDecision(p, p_lower, edge_lower, 0.0, 0.0, "LOWER_BOUND_EDGE_BELOW_MINIMUM")
    kelly = fractional_kelly_fraction(
        p_lower,
        price,
        cost_per_share,
        fraction=fraction,
    )
    return SizingDecision(p, p_lower, edge_lower, kelly, kelly, "KELLY_FROM_LOWER_BOUND")
