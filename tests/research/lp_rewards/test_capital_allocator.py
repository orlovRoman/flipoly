from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.capital_allocator import CapitalAllocator
from polyflip.research.lp_rewards.models import MarketPosition, OrderSide, VirtualOrder


def test_capital_allocator_limits():
    allocator = CapitalAllocator(
        allocated_working_capital=Decimal("100.00"),
        max_unhedged_per_market=Decimal("25.00"),
        max_unhedged_total=Decimal("50.00"),
    )

    positions = {}
    open_orders = {}

    # Order commitment of $20 on market 1 -> Allowed (<= $25, <= $50, <= $100)
    order_m1 = [
        VirtualOrder(
            order_id="o1",
            condition_id="m1",
            asset_id="tok1",
            side=OrderSide.BUY,
            price=Decimal("0.50"),
            size=Decimal("40.0"),  # $20.0
            placed_at_ns=0,
        )
    ]
    ok, reason = allocator.can_allocate_orders("m1", positions, open_orders, order_m1)
    assert ok is True

    # Order commitment of $30 on market 2 -> Rejected (> $25 per-market limit)
    order_m2_too_large = [
        VirtualOrder(
            order_id="o2",
            condition_id="m2",
            asset_id="tok2",
            side=OrderSide.BUY,
            price=Decimal("0.50"),
            size=Decimal("60.0"),  # $30.0
            placed_at_ns=0,
        )
    ]
    ok, reason = allocator.can_allocate_orders("m2", positions, open_orders, order_m2_too_large)
    assert ok is False
    assert "per-market" in reason


def test_r100_calendar_metric_and_buffer_normalization():
    allocator = CapitalAllocator(allocated_working_capital=Decimal("100.00"))

    # Case 1: Pure $100 capital, 7 full days, $28 Net PnL -> $4.00/day
    r100 = allocator.calculate_r100_calendar(net_pnl=Decimal("28.00"), completed_full_utc_days=7)
    assert r100 == Decimal("4.00")

    # Case 2: Extra $100 buffer used for 50% of the time (effective capital = $150)
    allocator.record_extra_buffer_usage(extra_buffer_amount=Decimal("100.00"), duration_seconds=Decimal("500.0"))
    allocator.record_extra_buffer_usage(extra_buffer_amount=Decimal("0.00"), duration_seconds=Decimal("500.0"))
    # Total duration = 1000s, extra time-weighted = 100 * 500 / 1000 = 50 -> effective capital = 150
    assert allocator.compute_effective_capital() == Decimal("150.00")

    # With $150 effective capital, $42 Net PnL in 7 days ($6.00/day on 150) -> normalized on $100 base is $4.00/day
    r100_buffered = allocator.calculate_r100_calendar(net_pnl=Decimal("42.00"), completed_full_utc_days=7)
    assert r100_buffered == Decimal("4.00")
