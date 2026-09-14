from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import (
    MarketRewardConfig,
    OrderSide,
    QuotingState,
    VirtualFill,
)
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM


def test_partial_fill_adjusts_opposite_leg():
    config = MarketRewardConfig(
        condition_id="c_partial",
        question="Partial Fill Test",
        rewards_daily_rate=Decimal("15.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("5.0"),
        oas=Decimal("5.0"),
        yes_token_id="tok_yes",
        no_token_id="tok_no",
    )
    fsm = MarketQuotingFSM(config, hedging_timeout_sec=900.0)

    # Place orders of size 30
    quotes = fsm.generate_quote_orders(midpoint=Decimal("0.50"), timestamp_ns=1000, target_size=Decimal("30.0"))
    yes_order = [q for q in quotes if q.asset_id == config.yes_token_id][0]
    no_order = [q for q in quotes if q.asset_id == config.no_token_id][0]

    # Partial fill of 12 shares on YES
    partial_fill = VirtualFill(
        fill_id="f_part",
        order_id=yes_order.order_id,
        condition_id=config.condition_id,
        asset_id=config.yes_token_id,
        side=OrderSide.BUY,
        price=yes_order.price,
        size=Decimal("12.0"),
        timestamp_ns=2000,
        queue_depletion_ratio=Decimal("0.4"),
    )

    fsm.on_fill(partial_fill)

    # State should be LONG_YES with 12 inventory
    assert fsm.state == QuotingState.LONG_YES
    assert fsm.position.yes_inventory == Decimal("12.0")

    # YES order remaining was cancelled; NO order resized to 12.0
    assert yes_order.order_id not in fsm.open_orders
    assert no_order.order_id in fsm.open_orders
    assert fsm.open_orders[no_order.order_id].size == Decimal("12.0")
