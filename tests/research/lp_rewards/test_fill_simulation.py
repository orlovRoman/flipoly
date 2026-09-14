from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.fill_simulation import QueuePositionTracker
from polyflip.research.lp_rewards.models import OrderSide, VirtualOrder


def test_oas_minimum_order_age_guard():
    order = VirtualOrder(
        order_id="ord_oas",
        condition_id="c1",
        asset_id="tok1",
        side=OrderSide.BUY,
        price=Decimal("0.50"),
        size=Decimal("20.0"),
        placed_at_ns=0,
    )
    # Market OAS is 5.0 seconds
    tracker = QueuePositionTracker(order=order, existing_depth_ahead=Decimal("0.0"), min_order_age_sec=Decimal("5.0"))

    # Trade arrives at 3.0 seconds (3 * 1e9 ns) -> should NOT fill
    fill_early = tracker.process_public_trade(
        trade_price=Decimal("0.50"),
        trade_size=Decimal("20.0"),
        trade_side=OrderSide.SELL,
        trade_timestamp_ns=3_000_000_000,
    )
    assert fill_early is None

    # Trade arrives at 5.5 seconds -> eligible and fills
    fill_eligible = tracker.process_public_trade(
        trade_price=Decimal("0.50"),
        trade_size=Decimal("20.0"),
        trade_side=OrderSide.SELL,
        trade_timestamp_ns=5_500_000_000,
    )
    assert fill_eligible is not None
    assert fill_eligible.size == Decimal("20.0")


def test_pessimistic_queue_depletion():
    order = VirtualOrder(
        order_id="ord_queue",
        condition_id="c1",
        asset_id="tok1",
        side=OrderSide.BUY,
        price=Decimal("0.50"),
        size=Decimal("10.0"),
        placed_at_ns=0,
    )
    # Queue ahead of us is 50.0 shares
    tracker = QueuePositionTracker(order=order, existing_depth_ahead=Decimal("50.0"), min_order_age_sec=Decimal("0.0"))

    # Trade of 30.0 shares at our price -> depletes queue from 50 to 20, no fill for us
    fill_1 = tracker.process_public_trade(
        trade_price=Decimal("0.50"),
        trade_size=Decimal("30.0"),
        trade_side=OrderSide.SELL,
        trade_timestamp_ns=1_000_000_000,
    )
    assert fill_1 is None
    assert tracker.depth_ahead == Decimal("20.0")

    # Second trade of 25.0 shares -> depletes remaining 20 queue, and fills 5 shares of our order
    fill_2 = tracker.process_public_trade(
        trade_price=Decimal("0.50"),
        trade_size=Decimal("25.0"),
        trade_side=OrderSide.SELL,
        trade_timestamp_ns=2_000_000_000,
    )
    assert fill_2 is not None
    assert fill_2.size == Decimal("5.0")
    assert tracker.depth_ahead == Decimal("0.0")
