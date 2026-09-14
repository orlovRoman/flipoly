from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import (
    MarketRewardConfig,
    OrderSide,
    QuotingState,
    VirtualFill,
)
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM


def test_complete_set_merge_and_pnl_realization():
    config = MarketRewardConfig(
        condition_id="c_merge",
        question="Complete Set Merge Test",
        rewards_daily_rate=Decimal("20.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        oas=Decimal("5.0"),
        yes_token_id="tok_yes",
        no_token_id="tok_no",
    )
    fsm = MarketQuotingFSM(config, hedging_timeout_sec=900.0)

    # 1. Place initial quotes
    quotes = fsm.generate_quote_orders(midpoint=Decimal("0.50"), timestamp_ns=1000, target_size=Decimal("25.0"))
    yes_order = [q for q in quotes if q.asset_id == config.yes_token_id][0]
    no_order = [q for q in quotes if q.asset_id == config.no_token_id][0]

    # Force specific prices: YES @ 0.48, NO @ 0.48 (sum = 0.96)
    yes_order.price = Decimal("0.48")
    no_order.price = Decimal("0.48")

    # 2. Fill YES leg (25 shares)
    fill_yes = VirtualFill(
        fill_id="f_yes",
        order_id=yes_order.order_id,
        condition_id=config.condition_id,
        asset_id=config.yes_token_id,
        side=OrderSide.BUY,
        price=Decimal("0.48"),
        size=Decimal("25.0"),
        timestamp_ns=2000,
        queue_depletion_ratio=Decimal("1.0"),
    )
    fsm.on_fill(fill_yes)
    assert fsm.state == QuotingState.LONG_YES
    assert fsm.position.yes_inventory == Decimal("25.0")

    # 3. Fill NO leg (25 shares) -> Triggers COMPLETE_SET -> MERGING -> FLAT
    fill_no = VirtualFill(
        fill_id="f_no",
        order_id=no_order.order_id,
        condition_id=config.condition_id,
        asset_id=config.no_token_id,
        side=OrderSide.BUY,
        price=Decimal("0.48"),
        size=Decimal("25.0"),
        timestamp_ns=3000,
        queue_depletion_ratio=Decimal("1.0"),
    )
    fsm.on_fill(fill_no)

    # After merge: state should be FLAT, inventories 0, realized PnL = (1.00 - 0.96) * 25 = $1.00
    assert fsm.state == QuotingState.FLAT
    assert fsm.position.yes_inventory == Decimal("0.0")
    assert fsm.position.no_inventory == Decimal("0.0")
    assert fsm.position.complete_sets_merged == Decimal("25.0")
    assert fsm.position.realized_trading_pnl == Decimal("1.00")
