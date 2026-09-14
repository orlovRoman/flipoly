from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import OrderbookLevel, OrderSide, VirtualOrder
from polyflip.research.lp_rewards.scoring import (
    calculate_competitor_and_own_scores,
    simulate_minute_monte_carlo,
)


def test_calculate_competitor_and_own_scores_normal():
    # Public book: midpoint is (0.49 + 0.51) / 2 = 0.50
    yes_bids = [OrderbookLevel(price=Decimal("0.49"), size=Decimal("100.0"))]
    yes_asks = [OrderbookLevel(price=Decimal("0.51"), size=Decimal("100.0"))]
    no_bids = [OrderbookLevel(price=Decimal("0.49"), size=Decimal("100.0"))]
    no_asks = [OrderbookLevel(price=Decimal("0.51"), size=Decimal("100.0"))]

    # Our orders: two-sided quotes inside max_spread (0.05)
    our_orders = [
        VirtualOrder(
            order_id="our_yes_bid",
            condition_id="c1",
            asset_id="tok_yes",
            side=OrderSide.BUY,
            price=Decimal("0.495"),
            size=Decimal("50.0"),
            placed_at_ns=1_700_000_000_000,
        ),
        VirtualOrder(
            order_id="our_yes_ask",
            condition_id="c1",
            asset_id="tok_yes",
            side=OrderSide.SELL,
            price=Decimal("0.505"),
            size=Decimal("50.0"),
            placed_at_ns=1_700_000_000_000,
        ),
    ]

    res = calculate_competitor_and_own_scores(
        condition_id="c1",
        timestamp_ns=1_700_000_000_000,
        public_yes_bids=yes_bids,
        public_yes_asks=yes_asks,
        public_no_bids=no_bids,
        public_no_asks=no_asks,
        our_orders=our_orders,
        max_spread=Decimal("0.05"),
        min_size=Decimal("10.0"),
        multiplier=Decimal("1.0"),
        yes_token_id="tok_yes",
        no_token_id="tok_no",
    )

    assert res.status == "VALID"
    assert res.p_mid_star == Decimal("0.50")
    assert res.q_own > Decimal("0.0")
    assert res.q_competitor_expected > Decimal("0.0")
    assert res.share_min <= res.share_expected <= res.share_max
    assert Decimal("0.0") < res.share_expected < Decimal("1.0")


def test_calculate_competitor_and_own_scores_uncertain():
    # Public book with no qualifying bids
    yes_bids = [OrderbookLevel(price=Decimal("0.49"), size=Decimal("5.0"))]  # < min_size 10
    yes_asks = [OrderbookLevel(price=Decimal("0.51"), size=Decimal("100.0"))]

    res = calculate_competitor_and_own_scores(
        condition_id="c1",
        timestamp_ns=1_700_000_000_000,
        public_yes_bids=yes_bids,
        public_yes_asks=yes_asks,
        public_no_bids=[],
        public_no_asks=[],
        our_orders=[],
        max_spread=Decimal("0.05"),
        min_size=Decimal("10.0"),
    )
    assert res.status == "MID_UNCERTAIN"
    assert res.p_mid_star is None
    assert res.share_expected == Decimal("0.0")


def test_simulate_minute_monte_carlo_reproducibility_and_bounds():
    snapshots = [
        {
            "condition_id": "c1",
            "timestamp_ns": 1_700_000_000_000,
            "yes_bids": [OrderbookLevel(price=Decimal("0.49"), size=Decimal("100.0"))],
            "yes_asks": [OrderbookLevel(price=Decimal("0.51"), size=Decimal("100.0"))],
            "no_bids": [OrderbookLevel(price=Decimal("0.49"), size=Decimal("100.0"))],
            "no_asks": [OrderbookLevel(price=Decimal("0.51"), size=Decimal("100.0"))],
            "yes_token_id": "tok_yes",
            "no_token_id": "tok_no",
        }
    ]

    our_orders = [
        VirtualOrder(
            order_id="our_yes_bid",
            condition_id="c1",
            asset_id="tok_yes",
            side=OrderSide.BUY,
            price=Decimal("0.495"),
            size=Decimal("50.0"),
            placed_at_ns=1_700_000_000_000,
        ),
        VirtualOrder(
            order_id="our_yes_ask",
            condition_id="c1",
            asset_id="tok_yes",
            side=OrderSide.SELL,
            price=Decimal("0.505"),
            size=Decimal("50.0"),
            placed_at_ns=1_700_000_000_000,
        ),
    ]

    res1 = simulate_minute_monte_carlo(
        snapshots=snapshots,
        our_orders=our_orders,
        daily_reward_pool=Decimal("100.0"),
        n_samples=50,
        seeds=(42, 123),
    )
    res2 = simulate_minute_monte_carlo(
        snapshots=snapshots,
        our_orders=our_orders,
        daily_reward_pool=Decimal("100.0"),
        n_samples=50,
        seeds=(42, 123),
    )

    # Deterministic across identical seeds
    assert res1["mean_daily_reward"] == res2["mean_daily_reward"]
    assert res1["ci_lower_95"] <= res1["mean_daily_reward"] <= res1["ci_upper_95"]
    assert res1["mean_daily_reward"] > Decimal("0.0")


def test_simulate_minute_monte_carlo_dust_threshold():
    snapshots = [
        {
            "condition_id": "c1",
            "timestamp_ns": 1_700_000_000_000,
            "yes_bids": [OrderbookLevel(price=Decimal("0.49"), size=Decimal("10000.0"))],
            "yes_asks": [OrderbookLevel(price=Decimal("0.51"), size=Decimal("10000.0"))],
            "no_bids": [OrderbookLevel(price=Decimal("0.49"), size=Decimal("10000.0"))],
            "no_asks": [OrderbookLevel(price=Decimal("0.51"), size=Decimal("10000.0"))],
            "yes_token_id": "tok_yes",
            "no_token_id": "tok_no",
        }
    ]

    # Very small order -> tiny share
    our_orders = [
        VirtualOrder(
            order_id="our_tiny_bid",
            condition_id="c1",
            asset_id="tok_yes",
            side=OrderSide.BUY,
            price=Decimal("0.49"),
            size=Decimal("10.0"),
            placed_at_ns=1_700_000_000_000,
        ),
        VirtualOrder(
            order_id="our_tiny_ask",
            condition_id="c1",
            asset_id="tok_yes",
            side=OrderSide.SELL,
            price=Decimal("0.51"),
            size=Decimal("10.0"),
            placed_at_ns=1_700_000_000_000,
        ),
    ]

    # Daily pool = 1.00 USDC, tiny share will yield reward < 1.00 USDC dust threshold
    res = simulate_minute_monte_carlo(
        snapshots=snapshots,
        our_orders=our_orders,
        daily_reward_pool=Decimal("1.00"),
        dust_threshold_usdc=Decimal("1.00"),
        n_samples=20,
        seeds=(42,),
    )
    assert res["mean_daily_reward"] == Decimal("0.0")
