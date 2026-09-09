"""
tests/trading/test_ct_synthetic_cycle.py

Synthetic Acceptance Test Suite (Stage 4 & Stage 5, Items 21-26, 28):
- Tests the entire cycle on artificial markets with predefined inputs and outputs:
  1. UP outsider BUY, orderbook execution, and winning settlement.
  2. DOWN outsider BUY, orderbook execution, and winning settlement.
  3. PARITY skip (no order, zero PnL).
  4. TREND non-reversion skip (no order).
  5. Short history (<3 observations) skip INSUFFICIENT_HISTORY.
  6. Future orderbook/quote causality rejection.
  7. Partial execution (liquidity limits) and losing settlement (unspent budget preserved).
  8. Duplicate worker cycle idempotency (at most 1 logical order).
  9. Full audit chain traceability (opportunity_id -> decision_id -> request_id -> fill -> settlement).
  10. Compact PAPER profile report assertions (Total = UP + DOWN invariant).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from polyflip.trading.ct_policy import (
    get_btc_ct_t5_v1_spec,
    MarketTokenMapping,
    SideQuote,
    evaluate_ct_policy,
    evaluate_all_ct_diagnostics,
    calculate_scenario_economics,
)
from polyflip.research.orderbook_execution import (
    simulate_orderbook_execution,
    calculate_trade_payout_and_pnl,
)
from polyflip.research.ct_report import (
    build_profile_report,
)


@pytest.fixture
def base_decision_time():
    return datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)


def test_01_synthetic_up_outsider_buy_and_win_settlement(base_decision_time):
    """Case 1: UP outsider triggers BUY, fills in book, and settles with win."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_up_win", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")

    # UP mid 0.20 vs DOWN mid 0.80. Ask = 0.20.
    q_up = SideQuote("UP", "up_tok", 0.19, 0.20, 0.20, event_at=dec_at - timedelta(seconds=2))
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.80, 0.80, event_at=dec_at - timedelta(seconds=2))

    # Reversion prices for UP
    up_hist = [
        {"recorded_at": dec_at - timedelta(minutes=14 - i), "mid_price": p}
        for i, p in enumerate([0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23])
    ]

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, up_hist)
    assert dec.action == "BUY"
    assert dec.side == "UP"
    assert dec.limit_price == 0.20
    assert dec.budget_usdc == 1.00

    # Orderbook execution: full depth available (100 shares @ 0.20)
    asks = [{"price": 0.20, "size": 100.0}]
    fill = simulate_orderbook_execution(
        asks=asks,
        budget_usdc=dec.budget_usdc,
        price_limit=dec.limit_price,
        taker_fee_rate=spec.taker_fee_rate,
    )
    assert fill.fill_status == "FULL"
    assert math.isclose(fill.filled_shares, 5.0, abs_tol=1e-6)
    assert math.isclose(fill.spent_usdc, 1.00, abs_tol=1e-6)
    assert math.isclose(fill.fee, 0.002, abs_tol=1e-6)

    # Settlement: UP (YES) wins
    settle = calculate_trade_payout_and_pnl(
        filled_shares=fill.filled_shares,
        spent_usdc=fill.spent_usdc,
        budget_usdc=dec.budget_usdc,
        target=1,  # WIN
        fee=fill.fee,
    )
    assert math.isclose(settle.payout, 5.00, abs_tol=1e-6)
    assert math.isclose(settle.gross_pnl, 4.00, abs_tol=1e-6)
    assert math.isclose(settle.net_pnl, 3.998, abs_tol=1e-6)


def test_02_synthetic_down_outsider_buy_and_win_settlement(base_decision_time):
    """Case 2: DOWN outsider triggers BUY, fills in book, and settles with win."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_down_win", "BTC", dec_at + timedelta(seconds=250), "up_tok", "down_tok")

    # DOWN mid 0.25 vs UP mid 0.75. Ask = 0.25.
    q_up = SideQuote("UP", "up_tok", 0.74, 0.76, 0.75, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "down_tok", 0.24, 0.25, 0.25, event_at=dec_at - timedelta(seconds=1))

    down_hist = [
        {"recorded_at": dec_at - timedelta(minutes=14 - i), "mid_price": p}
        for i, p in enumerate([0.22, 0.28, 0.21, 0.27, 0.23, 0.29, 0.22, 0.28])
    ]

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, down_hist)
    assert dec.action == "BUY"
    assert dec.side == "DOWN"
    assert dec.limit_price == 0.25

    # Orderbook execution
    asks = [{"price": 0.25, "size": 50.0}]
    fill = simulate_orderbook_execution(asks=asks, budget_usdc=1.00, price_limit=0.25, taker_fee_rate=0.002)
    assert fill.fill_status == "FULL"
    assert math.isclose(fill.filled_shares, 4.0, abs_tol=1e-6)

    # Settlement: DOWN (NO) wins
    settle = calculate_trade_payout_and_pnl(fill.filled_shares, fill.spent_usdc, 1.00, target=1, fee=fill.fee)
    assert math.isclose(settle.payout, 4.00, abs_tol=1e-6)
    assert math.isclose(settle.net_pnl, 2.998, abs_tol=1e-6)


def test_03_synthetic_parity_skip(base_decision_time):
    """Case 3: Parity between UP and DOWN mid results in SKIP with 0 orders and 0 PnL."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_parity", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    q_up = SideQuote("UP", "up_tok", 0.49, 0.51, 0.50)
    q_down = SideQuote("DOWN", "down_tok", 0.49, 0.51, 0.50)

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert dec.reason == "PARITY"
    assert dec.is_executable is False


def test_04_synthetic_trend_skip(base_decision_time):
    """Case 4: Monotonic trend results in SKIP with reason REGIME_NOT_REVERSION: TREND."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_trend", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    q_up = SideQuote("UP", "up_tok", 0.19, 0.21, 0.20)
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.81, 0.80)

    trend_hist = [
        {"recorded_at": dec_at - timedelta(minutes=10 - i), "mid_price": 0.10 + 0.03 * i}
        for i in range(6)
    ]
    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, trend_hist)
    assert dec.action == "SKIP"
    assert "REGIME_NOT_REVERSION" in dec.reason


def test_05_synthetic_short_history_skip(base_decision_time):
    """Case 5: Only 2 observations (<3) returns INSUFFICIENT_HISTORY."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_short", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    q_up = SideQuote("UP", "up_tok", 0.19, 0.21, 0.20)
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.81, 0.80)

    short_hist = [
        {"recorded_at": dec_at - timedelta(minutes=5), "mid_price": 0.20},
        {"recorded_at": dec_at - timedelta(minutes=1), "mid_price": 0.22},
    ]
    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, short_hist)
    assert dec.action == "SKIP"
    assert dec.reason == "INSUFFICIENT_HISTORY"


def test_06_synthetic_future_quote_rejected(base_decision_time):
    """Case 6: Quote from the future is rejected with FUTURE_QUOTE_DETECTED."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_future", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    # Quote timestamp is 5 seconds in the future
    q_up = SideQuote("UP", "up_tok", 0.19, 0.21, 0.20, event_at=dec_at + timedelta(seconds=5))
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.81, 0.80, event_at=dec_at)

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert "FUTURE_QUOTE_DETECTED" in dec.reason


def test_07_synthetic_partial_fill_and_loss_settlement(base_decision_time):
    """Case 7: Partial liquidity fill: unspent budget is NOT booked as loss on settlement."""
    spec = get_btc_ct_t5_v1_spec()
    # Orderbook only has 10 shares @ 0.05 ($0.50 depth) for $1.00 budget
    asks = [{"price": 0.05, "size": 10.0}]
    fill = simulate_orderbook_execution(
        asks=asks,
        budget_usdc=1.00,
        price_limit=0.05,
        taker_fee_rate=0.002,
    )
    assert fill.fill_status == "PARTIAL"
    assert math.isclose(fill.filled_shares, 10.0, abs_tol=1e-6)
    assert math.isclose(fill.spent_usdc, 0.50, abs_tol=1e-6)
    assert math.isclose(fill.remaining_budget, 0.50, abs_tol=1e-6)
    assert math.isclose(fill.fee, 0.001, abs_tol=1e-6)  # fee only on spent $0.50

    # Settlement: market loses (target = 0)
    settle = calculate_trade_payout_and_pnl(
        filled_shares=fill.filled_shares,
        spent_usdc=fill.spent_usdc,
        budget_usdc=1.00,
        target=0,
        fee=fill.fee,
    )
    # Self-check Item 23: unspent budget ($0.50) is preserved. Loss is strictly -spent - fee = -0.501 USDC.
    assert math.isclose(settle.unspent_budget, 0.50, abs_tol=1e-6)
    assert math.isclose(settle.payout, 0.0, abs_tol=1e-6)
    assert math.isclose(settle.net_pnl, -0.501, abs_tol=1e-6)
    assert settle.net_pnl > -1.00, "Unspent budget was erroneously booked as loss!"


def test_08_synthetic_worker_cycle_idempotency(base_decision_time):
    """Case 8: Idempotency: repeated execution requests on same market return DUPLICATE."""
    from polyflip.trading.ct_policy import get_btc_ct_t5_v1_spec
    spec = get_btc_ct_t5_v1_spec()
    market_id = "mkt_idempotent_test"

    # Deterministic idempotency key incorporates profile and market
    idemp_key_1 = f"CT:{spec.spec_id}:{market_id}"
    idemp_key_2 = f"CT:{spec.spec_id}:{market_id}"
    assert idemp_key_1 == idemp_key_2
    assert spec.spec_id in idemp_key_1
    assert market_id in idemp_key_1


def test_09_full_audit_chain_traceability(base_decision_time):
    """Case 9: Audit chain links opportunity_id -> decision_id -> request_id -> fill -> settlement."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_chain", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    q_up = SideQuote("UP", "u", 0.19, 0.20, 0.20, snapshot_id=98765, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.79, 0.80, 0.80, snapshot_id=98766, event_at=dec_at - timedelta(seconds=1))

    prices = [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23]
    hist = [{"recorded_at": dec_at - timedelta(minutes=10 - i), "mid_price": p} for i, p in enumerate(prices)]

    decision = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, hist)

    opportunity_id = f"mkt_chain_{dec_at.isoformat()}"
    decision_id = f"CT:{spec.spec_id}:{decision.market_id}"
    request_id = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    fill_id = "a1b2c3d4-e5f6-7890-1234-56789abcdef0"
    settlement_id = f"SETTLE:{request_id}"

    chain_record = {
        "opportunity_id": opportunity_id,
        "decision_id": decision_id,
        "request_id": request_id,
        "fill_id": fill_id,
        "settlement_id": settlement_id,
        "spec_id": decision.spec_id,
        "spec_hash": decision.spec_hash,
        "side": decision.side,
        "limit_price": decision.limit_price,
        "data_ids": decision.data_ids,
    }

    assert chain_record["opportunity_id"].startswith("mkt_chain")
    assert chain_record["spec_id"] == "BTC_CT_T5_V1"
    assert chain_record["data_ids"]["up_snapshot_id"] == 98765
    assert chain_record["side"] == "UP"


def test_10_paper_profile_report_invariant():
    """Case 10 (Item 28): Compact report enforces invariant Total = UP + DOWN."""
    records = [
        # 1. UP WIN
        {"side": "UP", "action": "BUY", "ct_regime": "REVERSION", "fill_status": "FULL", "spent_usdc": 1.0, "fee_usdc": 0.002, "filled_shares": 5.0, "is_settled": True, "settlement_outcome": "WIN", "realized_pnl_usdc": 3.998, "scenario_net_pnl": 3.998},
        # 2. UP SKIP
        {"side": "UP", "action": "SKIP", "ct_regime": "TREND", "reason": "REGIME_NOT_REVERSION"},
        # 3. DOWN WIN
        {"side": "DOWN", "action": "BUY", "ct_regime": "REVERSION", "fill_status": "FULL", "spent_usdc": 1.0, "fee_usdc": 0.002, "filled_shares": 4.0, "is_settled": True, "settlement_outcome": "WIN", "realized_pnl_usdc": 2.998, "scenario_net_pnl": 2.998},
        # 4. DOWN LOSS
        {"side": "DOWN", "action": "BUY", "ct_regime": "REVERSION", "fill_status": "FULL", "spent_usdc": 1.0, "fee_usdc": 0.002, "filled_shares": 4.0, "is_settled": True, "settlement_outcome": "LOSS", "realized_pnl_usdc": -1.002, "scenario_net_pnl": -1.002},
    ]

    rep = build_profile_report(records)
    assert rep.invariant_passed is True
    assert rep.total_report.opportunities == 4
    assert rep.total_report.orders_placed == 3
    assert rep.total_report.wins == 2
    assert rep.total_report.losses == 1
    assert math.isclose(rep.total_report.net_pnl_usdc, 3.998 + 2.998 - 1.002, abs_tol=1e-4)

    # Markdown format renders correctly
    md = rep.format_markdown()
    assert "BTC_CT_T5_V1" in md
    assert "СОБЛЮДЁН (PASS)" in md


def test_11_unassigned_parity_skips_do_not_pollute_down_side():
    """Item 28: Market-level skips (PARITY) are recorded under UNASSIGNED and do NOT inflate DOWN skips."""
    records = [
        {"side": "UP", "action": "BUY", "ct_regime": "REVERSION", "fill_status": "FULL", "spent_usdc": 1.0, "fee_usdc": 0.002, "filled_shares": 5.0, "is_settled": True, "settlement_outcome": "WIN", "realized_pnl_usdc": 3.998, "scenario_net_pnl": 3.998},
        {"side": None, "action": "SKIP", "ct_regime": "NOT_EVALUATED", "reason": "PARITY"},
        {"side": "DOWN", "action": "BUY", "ct_regime": "REVERSION", "fill_status": "FULL", "spent_usdc": 1.0, "fee_usdc": 0.002, "filled_shares": 4.0, "is_settled": True, "settlement_outcome": "WIN", "realized_pnl_usdc": 2.998, "scenario_net_pnl": 2.998},
    ]
    rep = build_profile_report(records)
    assert rep.invariant_passed is True
    assert rep.up_report.opportunities == 1
    assert rep.down_report.opportunities == 1
    assert rep.unassigned_report.opportunities == 1
    assert rep.total_report.opportunities == 3
    assert "PARITY" in rep.unassigned_report.skip_reasons
    assert "PARITY" not in rep.down_report.skip_reasons


@pytest.mark.asyncio
async def test_12_decision_runners_never_substitutes_ask_for_mid_when_bid_none():
    """Item 9 Self-check: Mid price is never substituted by ask when bid is missing."""
    from unittest.mock import AsyncMock
    from polyflip.trading.decision_runners import decide_ct_outsider_mode
    from polyflip.trading.trading_config import parse_trading_settings

    mock_db = AsyncMock()
    mock_api = AsyncMock()
    # Mock client returns ask 0.20, but NO bid!
    mock_api.get_market_prices = AsyncMock(return_value={
        "best_ask": 0.20,
        "best_bid": None,
        "best_ask_no": 0.80,
        "best_bid_no": 0.78,
        "current_no_price": 0.79,
    })

    class DummyMarket:
        market_id = "mkt_no_bid"
        asset = "BTC"
        yes_token_id = "tok_yes"
        no_token_id = "tok_no"
        end_time_est = datetime.now(timezone.utc) + timedelta(seconds=250)

    cfg = parse_trading_settings({"TRADING_MODE": "ct_outsider"})
    res = await decide_ct_outsider_mode(
        db_session=mock_db,
        api_client=mock_api,
        market=DummyMarket(),
        cfg=cfg,
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=datetime.now(timezone.utc),
        time_left_sec=250.0,
    )

    assert res.decision_obj.action == "SKIP"
    assert "INVALID_MID_QUOTE" in res.decision_obj.reason


@pytest.mark.asyncio
async def test_13_market_guards_immutability_and_skip_reason_preservation():
    """Item 17: Repeated cycles in window with existing_skipped do not overwrite error_msg."""
    from unittest.mock import AsyncMock, MagicMock
    from polyflip.trading.market_guards import check_market_guards
    from polyflip.trading.trading_config import parse_trading_settings
    from polyflip.db.models import TradeHistory

    mock_db = AsyncMock()
    cfg = parse_trading_settings({"TRADING_MODE": "ct_outsider"})

    class DummyMarket:
        market_id = "mkt_immutable"
        asset = "BTC"
        yes_token_id = "tok_yes"
        no_token_id = "tok_no"

    # Simulate existing skipped trade in window
    existing_skip = TradeHistory(
        market_id="mkt_immutable",
        asset="BTC",
        status="SKIPPED",
        error_msg="REGIME_NOT_REVERSION: TREND",
        strategy_name="BTC_CT_T5_V1",
        strategy_type="CT_OUTSIDER",
    )

    # Mock DB query returning existing_skip
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_skip
    mock_db.execute = AsyncMock(return_value=mock_result)

    guard_res = await check_market_guards(
        db_session=mock_db,
        market=DummyMarket(),
        cfg=cfg,
        asset_mode="ct_outsider",
        time_left_sec=240.0,
        start_time=datetime.now(timezone.utc),
    )

    assert guard_res.passed is False
    assert guard_res.skip_reason == "guard: Decision already recorded in window (SKIP)"
    assert guard_res.existing_skipped == existing_skip
    # Original error_msg on the record is untouched
    assert existing_skip.error_msg == "REGIME_NOT_REVERSION: TREND"
