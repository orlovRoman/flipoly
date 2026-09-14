from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.evaluation import (
    compute_block_bootstrap_ci,
    determine_gate_a_verdict,
)


def test_gate_a_all_five_verdicts():
    target = Decimal("3.50")

    # 1. EDGE_REJECTED: upper_95 <= 0
    v1 = determine_gate_a_verdict(
        point_est=Decimal("-1.20"),
        lower_95=Decimal("-2.50"),
        upper_95=Decimal("-0.10"),
        target_rate=target,
    )
    assert v1 == "EDGE_REJECTED"

    # 2. PROFITABLE_BELOW_TARGET: lower_95 > 0 and upper_95 < 3.50
    v2 = determine_gate_a_verdict(
        point_est=Decimal("1.80"),
        lower_95=Decimal("0.50"),
        upper_95=Decimal("2.90"),
        target_rate=target,
    )
    assert v2 == "PROFITABLE_BELOW_TARGET"

    # 3. TARGET_REJECTED: upper_95 < 3.50 (e.g., spanning zero but below target)
    v3 = determine_gate_a_verdict(
        point_est=Decimal("0.50"),
        lower_95=Decimal("-0.80"),
        upper_95=Decimal("1.80"),
        target_rate=target,
    )
    assert v3 == "TARGET_REJECTED"

    # 4. TARGET_PLAUSIBLE: lower_95 <= 3.50 <= upper_95
    v4 = determine_gate_a_verdict(
        point_est=Decimal("3.20"),
        lower_95=Decimal("1.10"),
        upper_95=Decimal("5.30"),
        target_rate=target,
    )
    assert v4 == "TARGET_PLAUSIBLE"

    # 5. PROCEED_LIVE: lower_95 > 0 and point_est >= 3.50 and total_days >= 7
    v5 = determine_gate_a_verdict(
        point_est=Decimal("4.20"),
        lower_95=Decimal("1.50"),
        upper_95=Decimal("6.80"),
        target_rate=target,
        total_days=7,
        min_days=7,
    )
    assert v5 == "PROCEED_LIVE"


def test_block_bootstrap_by_utc_day():
    # 7 days of strong profits on $100 base
    daily_pnls = [
        Decimal("4.00"),
        Decimal("3.80"),
        Decimal("4.20"),
        Decimal("3.50"),
        Decimal("4.50"),
        Decimal("3.90"),
        Decimal("4.10"),
    ]
    point, lower, upper = compute_block_bootstrap_ci(daily_pnls, n_bootstrap=1000, seed=42)

    assert point == Decimal("4.00")
    assert lower > Decimal("3.00")
    assert upper < Decimal("5.00")
