from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.fill_simulation import QueuePositionTracker
from polyflip.research.lp_rewards.models import OrderSide, VirtualOrder


def test_causality_trade_must_be_strictly_after_order():
    order = VirtualOrder(
        order_id="ord_causal",
        condition_id="c_causal",
        asset_id="tok1",
        side=OrderSide.BUY,
        price=Decimal("0.50"),
        size=Decimal("10.0"),
        placed_at_ns=1_000_000_000,  # Placed at t=1.0s
    )
    tracker = QueuePositionTracker(order=order, existing_depth_ahead=Decimal("0.0"), min_order_age_sec=Decimal("0.0"))

    # Trade from the past (t=0.5s) must be rejected
    past_fill = tracker.process_public_trade(
        trade_price=Decimal("0.50"),
        trade_size=Decimal("10.0"),
        trade_side=OrderSide.SELL,
        trade_timestamp_ns=500_000_000,
    )
    assert past_fill is None

    # Trade from future is accepted
    future_fill = tracker.process_public_trade(
        trade_price=Decimal("0.50"),
        trade_size=Decimal("10.0"),
        trade_side=OrderSide.SELL,
        trade_timestamp_ns=1_500_000_000,
    )
    assert future_fill is not None
    assert future_fill.timestamp_ns >= order.placed_at_ns
