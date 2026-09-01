"""Conservative uncertainty-aware position sizing for weighted policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
