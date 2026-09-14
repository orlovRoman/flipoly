from decimal import Decimal
from pathlib import Path
import pandas as pd
import pytest

from polyflip.research.lp_rewards.ledger import PortfolioLedger
from polyflip.research.lp_rewards.models import (
    MarketPosition,
    MarketRewardConfig,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderSide,
    QuotingState,
    VirtualFill,
)
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM


def test_single_taker_fee_deduction():
    config = MarketRewardConfig(
        condition_id="c_fee",
        question="Fee Question?",
        rewards_daily_rate=Decimal("10.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        taker_fee_rate=Decimal("0.01"),  # 1% taker fee
    )
    fsm = MarketQuotingFSM(config=config, hedging_timeout_sec=900.0)
    ledger = PortfolioLedger(allocated_capital=Decimal("100.00"))

    # Put FSM into LONG_YES with 100 shares bought at 0.50 ($50 cost)
    fsm.position.state = QuotingState.LONG_YES
    fsm.position.yes_inventory = Decimal("100.0")
    fsm.position.yes_fill_cost = Decimal("50.0")
    fsm.position.cash_invested = Decimal("50.0")
    fsm.state_timer_ns = 1_000_000_000_000

    # Book has bids: 100 shares at 0.52
    orderbook = OrderbookSnapshot(
        condition_id="c_fee",
        asset_id="tok_yes",
        bids=[OrderbookLevel(price=Decimal("0.52"), size=Decimal("100.0"))],
        asks=[],
        timestamp_ns=1_000_000_000_000 + 901 * 1_000_000_000,
        valid_from_ns=1_000_000_000_000,
    )

    # Trigger exit after 901s
    exited, fill = fsm.check_timeout_and_exit(1_000_000_000_000 + 901 * 1_000_000_000, orderbook)
    assert exited is True
    assert fill is not None

    # Gross revenue: 100 * 0.52 = 52.0
    # Cost basis: 50.0
    # Gross trading PnL: 52.0 - 50.0 = +2.0
    assert fsm.position.realized_trading_pnl == Decimal("2.0")

    # Taker fee: 52.0 * 0.01 = 0.52
    assert fill.taker_fee_paid == Decimal("0.52")
    assert fsm.position.taker_fees_paid == Decimal("0.52")

    # Record in ledger and verify net PnL has fee deducted once
    ledger.record_fill(fill)
    # Positions map
    positions = {config.condition_id: fsm.position}
    net_pnl = ledger.calculate_net_pnl(positions, executable_mtm=Decimal("0.0"))

    # Net PnL = realized_pnl (+2.0) - cash_invested(0) - total_taker_fees(0.52) = +1.48
    assert net_pnl == Decimal("1.48")


def test_partial_liquidity_exit_preserves_unhedged_position():
    config = MarketRewardConfig(
        condition_id="c_partial",
        question="Partial Exit Question?",
        rewards_daily_rate=Decimal("10.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        taker_fee_rate=Decimal("0.0"),
    )
    fsm = MarketQuotingFSM(config=config, hedging_timeout_sec=900.0)

    # 100 shares at 0.50 ($50 cost)
    fsm.position.state = QuotingState.LONG_YES
    fsm.position.yes_inventory = Decimal("100.0")
    fsm.position.yes_fill_cost = Decimal("50.0")
    fsm.position.cash_invested = Decimal("50.0")
    fsm.state_timer_ns = 1_000_000_000_000

    # Book has ONLY 40 shares of bids at 0.45
    orderbook = OrderbookSnapshot(
        condition_id="c_partial",
        asset_id="tok_yes",
        bids=[OrderbookLevel(price=Decimal("0.45"), size=Decimal("40.0"))],
        asks=[],
        timestamp_ns=1_000_000_000_000 + 901 * 1_000_000_000,
        valid_from_ns=1_000_000_000_000,
    )

    exited, fill = fsm.check_timeout_and_exit(1_000_000_000_000 + 901 * 1_000_000_000, orderbook)
    assert exited is True
    assert fill is not None
    assert fill.size == Decimal("40.0")

    # State must be EXIT_LIQUIDITY_INSUFFICIENT
    assert fsm.position.state == QuotingState.EXIT_LIQUIDITY_INSUFFICIENT
    assert fsm.exit_insufficient_liquidity_count == 1

    # Remaining 60 shares must be preserved with proportional cost
    assert fsm.position.yes_inventory == Decimal("60.0")
    assert fsm.position.yes_fill_cost == Decimal("30.0")

    # Sold 40 shares: revenue 40 * 0.45 = 18.0, cost 20.0 -> realized PnL = -2.0
    assert fsm.position.realized_trading_pnl == Decimal("-2.0")


def test_ledger_export_parquet_precision(tmp_path: Path):
    ledger = PortfolioLedger(allocated_capital=Decimal("100.00"))

    high_prec_fill = VirtualFill(
        fill_id="f_precise_1",
        order_id="o_precise_1",
        condition_id="c_prec",
        asset_id="tok_prec",
        side=OrderSide.BUY,
        price=Decimal("0.485123456789"),
        size=Decimal("123.456789123456"),
        timestamp_ns=1_700_000_000_123_456_789,
        queue_depletion_ratio=Decimal("0.8523"),
        taker_fee_paid=Decimal("0.0012345678"),
        markout_5s=Decimal("0.0051234"),
    )
    ledger.record_fill(high_prec_fill)

    out_file = tmp_path / "trades.parquet"
    ledger.export_trades_parquet(out_file)

    assert out_file.exists()
    df = pd.read_parquet(out_file)
    assert len(df) == 1

    # Price and size should be exact string representation preserved
    assert df.iloc[0]["price"] == "0.485123456789"
    assert df.iloc[0]["size"] == "123.456789123456"
    assert df.iloc[0]["queue_depletion_ratio"] == "0.8523"
    assert df.iloc[0]["taker_fee_paid"] == "0.0012345678"
    assert df.iloc[0]["markout_5s"] == "0.0051234"
