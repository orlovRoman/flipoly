from decimal import Decimal, getcontext
from typing import Dict, List, Tuple
import numpy as np

getcontext().prec = 28


def compute_block_bootstrap_ci(
    daily_net_pnls: List[Decimal],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[Decimal, Decimal, Decimal]:
    """Perform block bootstrap strictly by whole UTC days.
    
    Each sample in daily_net_pnls represents the aggregated Net PnL of all markets for one complete UTC day.
    Returns (point_estimate, lower_ci, upper_ci) for daily return on $100 allocated capital.
    """
    if not daily_net_pnls:
        return Decimal("0.0"), Decimal("0.0"), Decimal("0.0")

    pnl_array = np.array([float(p) for p in daily_net_pnls])
    point_est = float(np.mean(pnl_array))

    rng = np.random.default_rng(seed)
    n_days = len(pnl_array)
    resamples = rng.choice(pnl_array, size=(n_bootstrap, n_days), replace=True)
    resample_means = np.mean(resamples, axis=1)

    lower_bound = float(np.percentile(resample_means, 100 * (alpha / 2.0)))
    upper_bound = float(np.percentile(resample_means, 100 * (1.0 - alpha / 2.0)))

    return (
        Decimal(str(round(point_est, 4))),
        Decimal(str(round(lower_bound, 4))),
        Decimal(str(round(upper_bound, 4))),
    )


def determine_gate_a_verdict(
    point_est: Decimal,
    lower_95: Decimal,
    upper_95: Decimal,
    target_rate: Decimal = Decimal("3.50"),
    total_days: int = 7,
    min_days: int = 7,
) -> str:
    """Classify results into the 5 mutually exclusive Gate A verdicts:
    - EDGE_REJECTED: upper_95 <= 0
    - PROFITABLE_BELOW_TARGET: lower_95 > 0 and upper_95 < 3.50
    - TARGET_REJECTED: upper_95 < 3.50
    - TARGET_PLAUSIBLE: lower_95 <= 3.50 <= upper_95
    - PROCEED_LIVE: lower_95 > 0 and point_est >= 3.50 and total_days >= min_days
    """
    if upper_95 <= Decimal("0.0"):
        return "EDGE_REJECTED"

    if lower_95 > Decimal("0.0") and upper_95 < target_rate:
        return "PROFITABLE_BELOW_TARGET"

    if upper_95 < target_rate:
        return "TARGET_REJECTED"

    if lower_95 > Decimal("0.0") and point_est >= target_rate and total_days >= min_days:
        return "PROCEED_LIVE"

    if lower_95 <= target_rate <= upper_95:
        return "TARGET_PLAUSIBLE"

    return "TARGET_REJECTED"


def determine_gate_b_verdict(
    point_est: Decimal,
    lower_95: Decimal,
    upper_95: Decimal,
    max_drawdown: Decimal,
    mean_prediction_error: Decimal,
    target_rate: Decimal = Decimal("3.50"),
    max_drawdown_limit: Decimal = Decimal("0.10"),
    max_error_limit: Decimal = Decimal("0.30"),
) -> str:
    """Classify live results for Gate B:
    - TARGET_CONFIRMED: lower_95 >= 3.50, max_drawdown <= 0.10, prediction_error <= 0.30
    - TARGET_NOT_CONFIRMED: upper_95 < 3.50 or max_drawdown > 0.10
    - INCONCLUSIVE: CI spans across 3.50 threshold
    """
    if max_drawdown > max_drawdown_limit:
        return "TARGET_NOT_CONFIRMED"

    if upper_95 < target_rate:
        return "TARGET_NOT_CONFIRMED"

    if lower_95 >= target_rate and mean_prediction_error <= max_error_limit:
        return "TARGET_CONFIRMED"

    return "INCONCLUSIVE"
