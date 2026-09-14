from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import OrderbookLevel
from polyflip.research.lp_rewards.scoring import (
    calculate_level_score,
    calculate_q_components,
    calculate_daily_reward,
)


def test_level_score_and_cutoff():
    mid = Decimal("0.50")
    spread = Decimal("0.05")
    min_size = Decimal("10.0")

    # Order exactly at mid: distance = 0 -> spread_factor = 1.0 -> score = size
    score_at_mid = calculate_level_score(
        price=Decimal("0.50"),
        size=Decimal("100.0"),
        midpoint=mid,
        max_spread=spread,
        min_size=min_size,
    )
    assert score_at_mid == Decimal("100.0")

    # Order at boundary: distance = 0.05 -> score = 0
    score_at_bound = calculate_level_score(
        price=Decimal("0.55"),
        size=Decimal("100.0"),
        midpoint=mid,
        max_spread=spread,
        min_size=min_size,
    )
    assert score_at_bound == Decimal("0.0")

    # Order outside boundary: distance = 0.06 -> score = 0
    score_outside = calculate_level_score(
        price=Decimal("0.56"),
        size=Decimal("100.0"),
        midpoint=mid,
        max_spread=spread,
        min_size=min_size,
    )
    assert score_outside == Decimal("0.0")

    # Order below min_size: size = 5 < 10 -> score = 0
    score_small = calculate_level_score(
        price=Decimal("0.50"),
        size=Decimal("5.0"),
        midpoint=mid,
        max_spread=spread,
        min_size=min_size,
    )
    assert score_small == Decimal("0.0")


def test_q_components_inside_10_90_range():
    # Midpoint = 0.50 (inside [0.10, 0.90])
    mid = Decimal("0.50")
    spread = Decimal("0.05")
    min_size = Decimal("10.0")

    # Only YES bids provided (Q_one = 100, Q_two = 0)
    yes_bids = [OrderbookLevel(price=Decimal("0.50"), size=Decimal("100.0"))]
    yes_asks = []
    no_bids = []
    no_asks = []

    q_one, q_two, q_min = calculate_q_components(
        yes_bids, yes_asks, no_bids, no_asks, mid, spread, min_size
    )

    assert q_one == Decimal("100.0")
    assert q_two == Decimal("0.0")
    # In [0.10, 0.90]: Q_min = max(min(100, 0), max(100/3, 0/3)) = 100 / 3
    expected_q_min = Decimal("100.0") / Decimal("3.0")
    assert q_min == expected_q_min


def test_q_components_outside_10_90_range_strict_two_sided():
    # Midpoint = 0.05 (outside [0.10, 0.90])
    mid = Decimal("0.05")
    spread = Decimal("0.02")
    min_size = Decimal("10.0")

    # Only YES bids provided (Q_one = 100, Q_two = 0)
    yes_bids = [OrderbookLevel(price=Decimal("0.05"), size=Decimal("100.0"))]
    yes_asks = []
    no_bids = []
    no_asks = []

    q_one, q_two, q_min = calculate_q_components(
        yes_bids, yes_asks, no_bids, no_asks, mid, spread, min_size
    )

    assert q_one == Decimal("100.0")
    assert q_two == Decimal("0.0")
    # Outside [0.10, 0.90]: Q_min = min(100, 0) = 0.0 (one-sided strictly zeroed!)
    assert q_min == Decimal("0.0")


def test_daily_reward_dust_filter():
    daily_pool = Decimal("100.00")
    total_samples = 1440

    # High score yielding $5.00 -> kept
    q_epoch_high = Decimal("72.0")  # 72 / 1440 = 0.05 -> $5.00
    reward_high = calculate_daily_reward(q_epoch_high, total_samples, daily_pool, Decimal("1.00"))
    assert reward_high == Decimal("5.00")

    # Tiny score yielding $0.50 -> dust filtered to $0.00
    q_epoch_low = Decimal("7.2")  # 7.2 / 1440 = 0.005 -> $0.50
    reward_low = calculate_daily_reward(q_epoch_low, total_samples, daily_pool, Decimal("1.00"))
    assert reward_low == Decimal("0.00")
