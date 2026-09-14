from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import (
    MarketRewardConfig,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderSide,
    QuotingState,
    VirtualFill,
)
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM


def create_mock_config() -> MarketRewardConfig:
    return MarketRewardConfig(
        condition_id="c_test",
        question="Test Question?",
        rewards_daily_rate=Decimal("20.0"),
        rewards_max_spread=Decimal("0.04"),
        rewards_min_size=Decimal("10.0"),
        oas=Decimal("5.0"),
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        taker_fee_rate=Decimal("0.001"),  # 0.1% taker fee
    )


def test_fsm_quoting_both_and_yes_fill():
    config = create_mock_config()
    fsm = MarketQuotingFSM(config, hedging_timeout_sec=900.0, max_combined_fill_cost=Decimal("0.98"))

    # Initial state FLAT
    assert fsm.state == QuotingState.FLAT

    # Generate quotes at midpoint 0.50
    quotes = fsm.generate_quote_orders(midpoint=Decimal("0.50"), timestamp_ns=1_000_000_000, target_size=Decimal("20.0"))
    assert len(quotes) == 2
    assert fsm.state == QuotingState.QUOTING_BOTH

    # Simulate YES fill
    yes_order = [q for q in quotes if q.asset_id == config.yes_token_id][0]
    fill_yes = VirtualFill(
        fill_id="f1",
        order_id=yes_order.order_id,
        condition_id=config.condition_id,
        asset_id=config.yes_token_id,
        side=OrderSide.BUY,
        price=yes_order.price,
        size=Decimal("20.0"),
        timestamp_ns=2_000_000_000,
        queue_depletion_ratio=Decimal("1.0"),
    )

    cancelled = fsm.on_fill(fill_yes)
    assert fsm.state == QuotingState.LONG_YES
    assert fsm.position.yes_inventory == Decimal("20.0")

    # The opposite leg (BUY NO) remains open, and its price is constrained
    no_orders = [o for o in fsm.open_orders.values() if o.asset_id == config.no_token_id]
    assert len(no_orders) == 1
    assert no_orders[0].price + yes_order.price <= Decimal("0.98")


def test_fsm_timeout_forced_exit():
    config = create_mock_config()
    fsm = MarketQuotingFSM(config, hedging_timeout_sec=900.0)

    # Put into LONG_YES with 20 shares bought at 0.50
    fsm.position.state = QuotingState.LONG_YES
    fsm.position.yes_inventory = Decimal("20.0")
    fsm.position.yes_fill_cost = Decimal("10.0")  # 20 * 0.50
    fsm.position.cash_invested = Decimal("10.0")
    fsm.state_timer_ns = 1_000_000_000  # t = 1.0s

    # At t = 902s (> 900s timeout): Forced exit should trigger
    now_ns = (1 + 902) * 1_000_000_000

    orderbook = OrderbookSnapshot(
        condition_id=config.condition_id,
        asset_id=config.yes_token_id,
        bids=[
            OrderbookLevel(price=Decimal("0.48"), size=Decimal("15.0")),
            OrderbookLevel(price=Decimal("0.45"), size=Decimal("10.0")),
        ],
        asks=[],
        timestamp_ns=now_ns,
        valid_from_ns=now_ns,
    )

    exited, exit_fill = fsm.check_timeout_and_exit(now_ns, orderbook)
    assert exited is True
    assert exit_fill is not None
    assert exit_fill.size == Decimal("20.0")
    assert fsm.state == QuotingState.FLAT
    assert fsm.position.yes_inventory == Decimal("0.0")
    assert fsm.position.taker_fees_paid > Decimal("0.0")
