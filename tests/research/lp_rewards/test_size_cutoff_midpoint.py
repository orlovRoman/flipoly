from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import OrderbookLevel
from polyflip.research.lp_rewards.scoring import calculate_cutoff_midpoint, calculate_sample_scores


def test_cutoff_midpoint_filters_small_sizes():
    min_size = Decimal("10.0")

    # Bids: [0.50, size=2], [0.48, size=15]
    # Asks: [0.52, size=3], [0.54, size=20]
    # Filtered best bid = 0.48, best ask = 0.54 -> mid* = 0.51 (not 0.51 from 0.50/0.52)
    bids = [
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("2.0")),
        OrderbookLevel(price=Decimal("0.48"), size=Decimal("15.0")),
    ]
    asks = [
        OrderbookLevel(price=Decimal("0.52"), size=Decimal("3.0")),
        OrderbookLevel(price=Decimal("0.54"), size=Decimal("20.0")),
    ]

    mid_star = calculate_cutoff_midpoint(bids, asks, min_size)
    assert mid_star == Decimal("0.51")


def test_cutoff_midpoint_missing_side_returns_none_mid_uncertain():
    min_size = Decimal("10.0")

    # Bids have size >= 10, asks only have size < 10
    bids = [OrderbookLevel(price=Decimal("0.48"), size=Decimal("15.0"))]
    asks = [OrderbookLevel(price=Decimal("0.52"), size=Decimal("3.0"))]

    mid_star = calculate_cutoff_midpoint(bids, asks, min_size)
    assert mid_star is None

    # Verify calculate_sample_scores sets status to MID_UNCERTAIN
    score = calculate_sample_scores(
        condition_id="cond_1",
        timestamp_ns=1000,
        yes_bids=bids,
        yes_asks=asks,
        no_bids=[],
        no_asks=[],
        max_spread=Decimal("0.05"),
        min_size=min_size,
    )
    assert score.status == "MID_UNCERTAIN"
    assert score.p_mid_star is None
    assert score.q_min == Decimal("0.0")
