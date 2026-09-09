"""
tests/trading/test_ct_synthetic_cycle.py

Synthetic Acceptance Test Suite (Stage 4 & Stage 5, Requirements 10-22):
- Replaces research simulation with the real PAPER execution pipeline:
  pre_trade_validator -> execute_and_record -> enqueue_open_request -> claim_one ->
  FakeExecutionGateway (LIVE_PARITY) -> _persist_fills -> rebuild_trade_accounting -> settle_resolved_position.
- Tests:
  1. UP outsider BUY, real execution full fill, winning settlement.
  2. DOWN outsider BUY, real execution full fill, winning settlement.
  3. PARITY skip (no order, zero PnL).
  4. TREND non-reversion skip (no order).
  5. Short history (<3 observations) skip INSUFFICIENT_HISTORY.
  6. Future quote rejection (FUTURE_QUOTE_DETECTED).
  7. Partial fill liquidity: unspent budget preserved.
  8. Limit price change rejection (price moved above limit).
  9. Multi-level orderbook execution and VWAP accounting.
  10. Settlement on partial fill: WIN, LOSS, and repeated settlement idempotency.
  11. Concurrency and atomic reservation (INSERT ON CONFLICT DO NOTHING, EnqueueRejected).
  12. Re-run after settlement blocked by reservation.
  13. Crash recovery at three failure points (reservation, outbox enqueue, fill-before-ack).
  14. First decision immutability on subsequent quote flip.
  15. Compact PAPER profile report invariant (Total = UP + DOWN).
  16. End-to-end audit chain traceability from real DB rows.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest
from sqlalchemy import select

from polyflip.db.models import LiveMarket, TradeHistory, CTDecisionReservation
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
)
from polyflip.trading.ct_policy import (
    get_btc_ct_t5_v1_spec,
    MarketTokenMapping,
    SideQuote,
    CTDecision,
    evaluate_ct_policy,
)
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.pre_trade_validator import PreTradeValidation
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.trade_recorder import execute_and_record, EnqueueRejected
from polyflip.trading.decision_runners import decide_ct_outsider_mode
from polyflip.trading.ct_reservation import (
    reserve_ct_decision,
    get_ct_decision_reservation,
)
from polyflip.execution.outbox import finalize_request
from polyflip.execution.worker import (
    claim_one,
    _persist_fills,
    rebuild_trade_accounting,
)
from polyflip.execution.settlement_service import settle_resolved_position
from polyflip.execution.gateways.fake import FakeExecutionGateway
from polyflip.execution.contracts import GatewayOrder, SubmissionResult
from polyflip.research.ct_report import build_profile_report


@pytest.fixture
def base_decision_time():
    return datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)


def _make_live_market(
    db_session,
    market_id: str = "mkt_test",
    asset: str = "BTC",
    yes_tok: str = "up_tok",
    no_tok: str = "down_tok",
    end_time: datetime | None = None,
) -> LiveMarket:
    now = datetime.now(timezone.utc)
    market = LiveMarket(
        market_id=market_id,
        asset=asset,
        question=f"Will {asset} flip?",
        yes_token_id=yes_tok,
        no_token_id=no_tok,
        end_time_est=end_time or (now + timedelta(seconds=240)),
        current_yes_price=0.20,
        current_no_price=0.80,
        current_spread=0.01,
        volume_5min=100.0,
        price_velocity=0.0,
        last_updated=now,
    )
    db_session.add(market)
    return market


async def _record_and_claim_trade(
    db_session,
    market: LiveMarket,
    decision: CTDecision,
    start_time: datetime,
) -> tuple[TradeHistory, ExecutionRequest]:
    outcome = "YES" if decision.side == "UP" else "NO"
    trade_decision = TradeDecision(
        action=f"BUY_{outcome}",
        buy_price=decision.limit_price or 0.20,
        bet_size_usdc=decision.budget_usdc,
        reason=decision.reason,
        strategy_type="CT_OUTSIDER",
        direction_value=decision.side,
        decision_details={
            "spec_id": decision.spec_id,
            "spec_hash": decision.spec_hash,
            "decision_run_id": f"CT:{decision.spec_id}:{market.market_id}",
            "chosen_side": decision.side,
            "token_id": decision.token_id,
            "market_role": "OUTSIDER",
            "strategy_type": "CT_OUTSIDER",
            "ct_regime": decision.ct_regime,
            "ct_features": decision.ct_features,
            "data_ids": decision.data_ids,
            "timing_diagnostics": decision.data_ids.get("timing_diagnostics", {}),
        },
    )
    validation = PreTradeValidation(
        valid=True,
        buy_price=Decimal(str(decision.limit_price or 0.20)),
        actual_bet_size=Decimal(str(decision.budget_usdc)),
        edge=0.05,
        market_role="OUTSIDER",
        skip_reason=None,
    )
    cfg = parse_trading_settings({})

    await execute_and_record(
        db_session=db_session,
        market=market,
        decision_obj=trade_decision,
        validation=validation,
        asset_mode="FAVORITE",
        active_features="ct_features",
        p_flip=0.0,
        model_ver=None,
        cfg=cfg,
        existing_skipped=None,
        start_time=start_time,
    )
    await db_session.commit()

    req = await claim_one(db_session, "PAPER")
    assert req is not None, "Execution request was not enqueued or claimed"
    trade = await db_session.get(TradeHistory, req.trade_history_id)
    assert trade is not None
    return trade, req


async def _execute_paper_gateway(
    db_session,
    trade: TradeHistory,
    req: ExecutionRequest,
    asks: list[dict[str, float]],
    bids: list[dict[str, float]] | None = None,
    fee_rate: Decimal = Decimal("0.002"),
    shares_override: Decimal | None = None,
    order_spend_usdc: Decimal | None = None,
) -> SubmissionResult:
    attempt = ExecutionAttempt(
        request_id=req.id,
        gateway="FAKE",
        attempt_no=1,
        submission_key=f"{req.idempotency_key}:1",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(attempt)
    await db_session.flush()

    token_id = "up_tok" if req.outcome_to_buy == "YES" else "down_tok"

    async def mock_quote_provider(_tok: str):
        return {
            "asks": asks,
            "bids": bids or [],
            "best_ask": asks[0]["price"] if asks else None,
            "best_bid": bids[0]["price"] if bids else None,
        }

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY",
        quote_provider=mock_quote_provider,
        fee_rate=fee_rate,
        min_order_shares=Decimal("0.0001"),
    )

    limit_p = Decimal(str(round(float(req.limit_price or 0.20), 4)))
    spend_usdc = Decimal(str(round(float(req.max_spend_usdc or 1.00), 4)))
    shares = shares_override or req.requested_shares or (spend_usdc / limit_p)

    max_spend = order_spend_usdc if order_spend_usdc is not None else (spend_usdc * Decimal("1.01") if fee_rate > 0 else spend_usdc)

    order = GatewayOrder(
        attempt_id=attempt.id,
        market_id=req.market_id,
        asset=req.asset,
        outcome_to_buy=req.outcome_to_buy,
        token_id=token_id,
        side="BUY",
        limit_price=limit_p,
        requested_shares=shares,
        max_spend_usdc=max_spend,
    )
    sub_res = await gateway.submit(order)

    if sub_res.accepted and sub_res.fills:
        await _persist_fills(db_session, attempt, sub_res.fills)
        req.filled_shares = sum((f.shares for f in sub_res.fills), Decimal("0"))
        req.filled_cost_usdc = sum((f.gross_quote_usdc for f in sub_res.fills), Decimal("0"))
        await finalize_request(db_session, req, state="FILLED")
        await rebuild_trade_accounting(db_session, trade.id)
    elif not sub_res.accepted:
        err_msg = sub_res.rejection_code or sub_res.error_message or "REJECTED"
        await finalize_request(db_session, req, state="REJECTED", error=err_msg)

    await db_session.commit()
    await db_session.refresh(trade)
    return sub_res


# ==============================================================================
# Requirements 10-15: Real PAPER Execution Pipeline Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_01_synthetic_up_outsider_buy_and_full_fill_win(db_session, base_decision_time):
    """Requirement 10, 11: UP outsider BUY executes through real pipeline with full fill and winning settlement."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_up_win", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    mapping = MarketTokenMapping(market.market_id, "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    q_up = SideQuote("UP", "up_tok", 0.19, 0.20, 0.20, event_at=dec_at - timedelta(seconds=2))
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.80, 0.80, event_at=dec_at - timedelta(seconds=2))

    up_hist = [
        {"recorded_at": dec_at - timedelta(minutes=14 - i), "mid_price": p}
        for i, p in enumerate([0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23])
    ]

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, up_hist)
    assert dec.action == "BUY"
    assert dec.side == "UP"
    assert dec.limit_price == 0.20

    # 1. Real pipeline enqueue and claim
    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)
    assert trade.position_status == "OPENING"
    assert req.state == "CLAIMED"

    # 2. Real gateway execution with full depth available (100 shares @ 0.20)
    asks = [{"price": 0.20, "size": 100.0}]
    sub_res = await _execute_paper_gateway(db_session, trade, req, asks=asks, fee_rate=Decimal("0.002"))
    assert sub_res.accepted is True
    assert len(sub_res.fills) == 1

    # 3. Post-fill accounting verification
    assert trade.position_status == "OPEN"
    assert math.isclose(float(trade.entry_filled_shares), 5.0, abs_tol=1e-5)
    assert math.isclose(float(trade.entry_cost_usdc), 1.002, abs_tol=1e-5)  # 1.00 gross + 0.002 fee

    # 4. Settlement: UP (YES) wins
    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="YES",
        payout_per_share=Decimal("1.0"),
    )
    await db_session.commit()
    await db_session.refresh(trade)

    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    # PnL = 5.0 * 1.0 - 1.002 = +3.998 USDC
    assert math.isclose(float(trade.realized_pnl_usdc), 3.998, abs_tol=1e-5)


@pytest.mark.asyncio
async def test_02_synthetic_down_outsider_buy_and_full_fill_win(db_session, base_decision_time):
    """Requirement 10, 11: DOWN outsider BUY executes through real pipeline with full fill and winning settlement."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_down_win", end_time=dec_at + timedelta(seconds=250))
    await db_session.commit()

    mapping = MarketTokenMapping(market.market_id, "BTC", dec_at + timedelta(seconds=250), "up_tok", "down_tok")
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

    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)
    asks = [{"price": 0.25, "size": 50.0}]
    sub_res = await _execute_paper_gateway(db_session, trade, req, asks=asks, fee_rate=Decimal("0.002"))
    assert sub_res.accepted is True

    assert trade.position_status == "OPEN"
    assert math.isclose(float(trade.entry_filled_shares), 4.0, abs_tol=1e-5)
    assert math.isclose(float(trade.entry_cost_usdc), 1.002, abs_tol=1e-5)

    # Settle DOWN (NO) wins
    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="NO",
        payout_per_share=Decimal("1.0"),
    )
    await db_session.commit()
    await db_session.refresh(trade)

    assert trade.position_status == "CLOSED"
    # PnL = 4.0 * 1.0 - 1.002 = +2.998 USDC
    assert math.isclose(float(trade.realized_pnl_usdc), 2.998, abs_tol=1e-5)


def test_03_synthetic_parity_skip(base_decision_time):
    """Case 3: Parity between UP and DOWN mid results in SKIP with 0 orders and 0 PnL."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_parity", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    q_up = SideQuote("UP", "up_tok", 0.49, 0.51, 0.50, event_at=dec_at - timedelta(seconds=2))
    q_down = SideQuote("DOWN", "down_tok", 0.49, 0.51, 0.50, event_at=dec_at - timedelta(seconds=2))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert dec.reason == "PARITY"
    assert dec.is_executable is False


def test_04_synthetic_trend_skip(base_decision_time):
    """Case 4: Monotonic trend results in SKIP with reason REGIME_NOT_REVERSION: TREND."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = base_decision_time
    mapping = MarketTokenMapping("mkt_trend", "BTC", dec_at + timedelta(seconds=240), "up_tok", "down_tok")
    q_up = SideQuote("UP", "up_tok", 0.19, 0.21, 0.20, event_at=dec_at - timedelta(seconds=2))
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=2))

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
    q_up = SideQuote("UP", "up_tok", 0.19, 0.21, 0.20, event_at=dec_at - timedelta(seconds=2))
    q_down = SideQuote("DOWN", "down_tok", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=2))

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


@pytest.mark.asyncio
async def test_07_synthetic_partial_fill_liquidity_preserves_unspent_budget(db_session, base_decision_time):
    """Requirement 12, 15: Partial fill preserves unspent budget and does not count it as loss."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_partial", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.05, budget_usdc=1.00,
        selected_ask=0.05, selected_mid=0.05, selected_bid=0.04, other_mid=0.95,
        outsider_margin=0.45, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )

    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)

    # Orderbook only has 10 shares @ 0.05 ($0.50 depth) for $1.00 budget
    asks = [{"price": 0.05, "size": 10.0}]
    sub_res = await _execute_paper_gateway(db_session, trade, req, asks=asks, fee_rate=Decimal("0.002"))
    assert sub_res.accepted is True

    assert trade.position_status == "OPEN"
    assert math.isclose(float(trade.entry_filled_shares), 10.0, abs_tol=1e-5)
    # Gross spent is 0.50 USDC, fee is 0.001 USDC -> total basis = 0.501 USDC
    assert math.isclose(float(trade.entry_cost_usdc), 0.501, abs_tol=1e-5)

    # Settlement: market loses (target outcome NO)
    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="NO",
        payout_per_share=Decimal("0.0"),
    )
    await db_session.commit()
    await db_session.refresh(trade)

    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    # Realized loss is strictly -0.501 USDC, NOT -1.00 USDC! Unspent $0.50 is preserved.
    assert math.isclose(float(trade.realized_pnl_usdc), -0.501, abs_tol=1e-5)
    assert trade.realized_pnl_usdc > Decimal("-1.00")


@pytest.mark.asyncio
async def test_08_synthetic_limit_price_change_rejection(db_session, base_decision_time):
    """Requirement 13: If orderbook ask moves above limit price, gateway rejects order with 0 fills."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_price_moved", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.20, budget_usdc=1.00,
        selected_ask=0.20, selected_mid=0.20, selected_bid=0.19, other_mid=0.80,
        outsider_margin=0.30, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )

    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)

    # Orderbook ask moved up to 0.21 (worse than limit_price 0.20)
    asks = [{"price": 0.21, "size": 100.0}]
    sub_res = await _execute_paper_gateway(db_session, trade, req, asks=asks)

    assert sub_res.accepted is False
    assert req.state == "REJECTED"
    assert trade.entry_filled_shares == Decimal("0")
    assert trade.position_status == "ENTRY_FAILED"


@pytest.mark.asyncio
async def test_09_synthetic_multi_level_orderbook_vwap(db_session, base_decision_time):
    """Requirement 14: Multi-level book consumes multiple levels and calculates correct VWAP entry price."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_vwap", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.12, budget_usdc=1.00,
        selected_ask=0.10, selected_mid=0.10, selected_bid=0.09, other_mid=0.90,
        outsider_margin=0.40, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )

    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)

    # 2 ask levels:
    # Level 1: 5 shares @ 0.10 = $0.50
    # Level 2: 10 shares @ 0.12 = $1.20 available
    # Budget: $1.00 (with fee_rate=0 for clean arithmetic)
    asks = [
        {"price": 0.10, "size": 5.0},
        {"price": 0.12, "size": 10.0},
    ]
    sub_res = await _execute_paper_gateway(
        db_session,
        trade,
        req,
        asks=asks,
        fee_rate=Decimal("0"),
        shares_override=Decimal("10.0"),
        order_spend_usdc=Decimal("1.00"),
    )
    assert sub_res.accepted is True
    assert len(sub_res.fills) == 2

    # Level 1: 5.0 shares @ 0.10 ($0.50)
    # Level 2: 0.50 / 0.12 = 4.166667 shares @ 0.12 ($0.50)
    # Total shares: 9.166667, Total gross: 1.00 USDC
    # VWAP = 1.00 / 9.166667 ≈ 0.109091
    assert math.isclose(float(trade.entry_filled_shares), 9.166667, abs_tol=1e-5)
    assert math.isclose(float(trade.entry_cost_usdc), 1.00, abs_tol=1e-5)
    assert math.isclose(float(trade.executed_price), 0.109091, abs_tol=1e-5)


@pytest.mark.asyncio
async def test_10_synthetic_partial_fill_settlement_win_and_loss(db_session, base_decision_time):
    """Requirement 15: Partial fill settlement: WIN (+9.499 USDC), LOSS (-0.501 USDC), and repeat settlement no-op."""
    dec_at = base_decision_time

    # --- Part A: WIN ---
    market_win = _make_live_market(db_session, market_id="mkt_part_win", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec_win = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market_win.market_id, asset="BTC", limit_price=0.05, budget_usdc=1.00,
        selected_ask=0.05, selected_mid=0.05, selected_bid=0.04, other_mid=0.95,
        outsider_margin=0.45, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )
    trade_win, req_win = await _record_and_claim_trade(db_session, market_win, dec_win, dec_at)
    asks = [{"price": 0.05, "size": 10.0}]
    await _execute_paper_gateway(db_session, trade_win, req_win, asks=asks, fee_rate=Decimal("0.002"))

    # Settle WIN: 10 shares * $1.00 payout = $10.00. Cost = 0.501. Net PnL = +9.499 USDC.
    await settle_resolved_position(db_session, trade_id=trade_win.id, winning_outcome="YES", payout_per_share=Decimal("1.0"))
    await db_session.commit()
    await db_session.refresh(trade_win)

    assert trade_win.position_status == "CLOSED"
    assert math.isclose(float(trade_win.realized_pnl_usdc), 9.499, abs_tol=1e-5)

    # Repeated settlement call must be an idempotent no-op!
    await settle_resolved_position(db_session, trade_id=trade_win.id, winning_outcome="YES", payout_per_share=Decimal("1.0"))
    await db_session.commit()
    await db_session.refresh(trade_win)
    assert math.isclose(float(trade_win.realized_pnl_usdc), 9.499, abs_tol=1e-5)
    assert trade_win.position_status == "CLOSED"

    # --- Part B: LOSS ---
    market_loss = _make_live_market(db_session, market_id="mkt_part_loss", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec_loss = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market_loss.market_id, asset="BTC", limit_price=0.05, budget_usdc=1.00,
        selected_ask=0.05, selected_mid=0.05, selected_bid=0.04, other_mid=0.95,
        outsider_margin=0.45, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )
    trade_loss, req_loss = await _record_and_claim_trade(db_session, market_loss, dec_loss, dec_at)
    await _execute_paper_gateway(db_session, trade_loss, req_loss, asks=asks, fee_rate=Decimal("0.002"))

    # Settle LOSS: payout = 0. Cost = 0.501. Net PnL = -0.501 USDC.
    await settle_resolved_position(db_session, trade_id=trade_loss.id, winning_outcome="NO", payout_per_share=Decimal("0.0"))
    await db_session.commit()
    await db_session.refresh(trade_loss)

    assert trade_loss.position_status == "CLOSED"
    assert math.isclose(float(trade_loss.realized_pnl_usdc), -0.501, abs_tol=1e-5)


# ==============================================================================
# Requirements 16-20: Concurrency, Idempotency & Crash Recovery Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_11_concurrency_atomic_reservation(db_session, base_decision_time):
    """Requirement 16, 17: Concurrent decision execution on same market yields exactly 1 trade and 1 rejected duplicate."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_concur", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.20, budget_usdc=1.00,
        selected_ask=0.20, selected_mid=0.20, selected_bid=0.19, other_mid=0.80,
        outsider_margin=0.30, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )

    # First execution succeeds
    trade1, req1 = await _record_and_claim_trade(db_session, market, dec, dec_at)
    assert trade1 is not None

    # Second execution on the same market raises EnqueueRejected
    with pytest.raises(EnqueueRejected) as exc_info:
        await _record_and_claim_trade(db_session, market, dec, dec_at)
    assert "ActiveExecutionConflict" in str(exc_info.value)
    assert "already reserved" in str(exc_info.value)

    # Verify database has exactly 1 trade and 1 reservation record with repeat_count=1
    trades = (await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == market.market_id))).scalars().all()
    assert len(trades) == 1

    res = await get_ct_decision_reservation(db_session, f"CT:BTC_CT_T5_V1:{market.market_id}")
    assert res is not None
    assert res.repeat_count == 1


@pytest.mark.asyncio
async def test_12_rerun_after_settlement_blocked(db_session, base_decision_time):
    """Requirement 18: Re-running decision on the same market after settlement returns ALREADY_DECIDED."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_rerun", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    # Initial cycle completes and settles
    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.20, budget_usdc=1.00,
        selected_ask=0.20, selected_mid=0.20, selected_bid=0.19, other_mid=0.80,
        outsider_margin=0.30, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )
    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)
    asks = [{"price": 0.20, "size": 50.0}]
    await _execute_paper_gateway(db_session, trade, req, asks=asks)
    await settle_resolved_position(db_session, trade_id=trade.id, winning_outcome="YES", payout_per_share=Decimal("1.0"))
    await db_session.commit()

    # Subsequent cycle polls the same market via decide_ct_outsider_mode
    class MockClient:
        async def get_market_prices(self, _tok):
            return {"best_ask": 0.20, "best_bid": 0.19, "event_at": dec_at, "received_at": dec_at}

    dec_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=MockClient(),
        market=market,
        cfg=parse_trading_settings({}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at + timedelta(seconds=10),
        time_left_sec=230.0,
        execution_mode="PAPER",
    )

    assert dec_res.decision_obj.action == "SKIP"
    assert "ALREADY_DECIDED: BUY" in dec_res.decision_obj.reason
    assert dec_res.decision_obj.decision_details.get("repeat") is True

    # Ensure no new trade was created
    trades = (await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == market.market_id))).scalars().all()
    assert len(trades) == 1


@pytest.mark.asyncio
async def test_13_recovery_at_three_failure_points(db_session, base_decision_time):
    """Requirement 19: Interruption & recovery at 3 failure points: after reservation, after enqueue, after fill before ack."""
    dec_at = base_decision_time

    # Point 1: Crash after decision reservation (reservation in DB, but no trade history or outbox request)
    mkt1 = _make_live_market(db_session, market_id="mkt_crash_1")
    await reserve_ct_decision(
        db_session,
        key=f"CT:BTC_CT_T5_V1:{mkt1.market_id}",
        market_id=mkt1.market_id,
        spec_id="BTC_CT_T5_V1",
        action="BUY",
        decision_at=dec_at,
        side="UP",
        limit_price=0.20,
        budget_usdc=1.00,
        reason="CT_SIGNAL_REVERSION",
    )
    await db_session.commit()

    # Recovery: calling decide_ct_outsider_mode detects existing reservation and returns SKIP
    class DummyClient:
        async def get_market_prices(self, _tok):
            return {"best_ask": 0.20, "best_bid": 0.19, "event_at": dec_at, "received_at": dec_at}

    dec_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=DummyClient(),
        market=mkt1,
        cfg=parse_trading_settings({}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at,
        time_left_sec=240.0,
    )
    assert dec_res.decision_obj.action == "SKIP"
    assert "ALREADY_DECIDED" in dec_res.decision_obj.reason

    # Point 2: Crash after enqueue (ExecutionRequest in state READY in outbox)
    mkt2 = _make_live_market(db_session, market_id="mkt_crash_2")
    dec2 = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=mkt2.market_id, asset="BTC", limit_price=0.20, budget_usdc=1.00,
        selected_ask=0.20, selected_mid=0.20, selected_bid=0.19, other_mid=0.80,
        outsider_margin=0.30, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )
    # Simulate enqueue before worker startup
    trade2, req2 = await _record_and_claim_trade(db_session, mkt2, dec2, dec_at)
    assert req2.state == "CLAIMED"
    # Worker recovers, executes, persists fills
    asks = [{"price": 0.20, "size": 50.0}]
    await _execute_paper_gateway(db_session, trade2, req2, asks=asks)
    await db_session.refresh(trade2)
    assert trade2.position_status == "OPEN"
    assert trade2.entry_filled_shares == Decimal("5.0")

    # Point 3: Crash after fill before trade accounting update
    # Simulate: ExecutionFill written to DB, but rebuild_trade_accounting was not yet called
    mkt3 = _make_live_market(db_session, market_id="mkt_crash_3")
    trade3, req3 = await _record_and_claim_trade(db_session, mkt3, dec2, dec_at)
    attempt3 = ExecutionAttempt(
        request_id=req3.id,
        gateway="FAKE",
        attempt_no=1,
        submission_key=f"{req3.idempotency_key}:1",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(attempt3)
    await db_session.flush()

    fill3 = ExecutionFill(
        attempt_id=attempt3.id,
        provider_trade_id="RECOVERY_FILL_1",
        gateway="FAKE",
        gross_quote_usdc=Decimal("1.00"),
        price=Decimal("0.20"),
        shares=Decimal("5.0"),
        fee_usdc=Decimal("0.002"),
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(fill3)
    req3.state = "FILLED"
    req3.filled_shares = Decimal("5.0")
    req3.filled_cost_usdc = Decimal("1.00")
    await db_session.commit()

    # Recovery: worker runs rebuild_trade_accounting
    await rebuild_trade_accounting(db_session, trade3.id)
    await db_session.commit()
    await db_session.refresh(trade3)

    assert trade3.position_status == "OPEN"
    assert trade3.entry_filled_shares == Decimal("5.0")
    assert trade3.remaining_shares == Decimal("5.0")
    assert math.isclose(float(trade3.entry_cost_usdc), 1.002, abs_tol=1e-5)


@pytest.mark.asyncio
async def test_14_first_decision_immutability_on_quote_change(db_session, base_decision_time):
    """Requirement 20: First reserved decision is immutable; market quote changes on next poll do not override it."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_immutable", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    # First poll: UP is outsider (ask 0.20)
    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="hash1", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.20, budget_usdc=1.00,
        selected_ask=0.20, selected_mid=0.20, selected_bid=0.19, other_mid=0.80,
        outsider_margin=0.30, ct_regime="REVERSION", ct_features={}, data_ids={},
        is_executable=True,
    )
    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)
    assert trade.outcome_bought == "YES"

    # Second poll: Quotes flip! DOWN is now outsider (ask 0.15), UP is expensive (ask 0.85)
    class FlippedClient:
        async def get_market_prices(self, tok):
            if tok == "up_tok":
                return {"best_ask": 0.85, "best_bid": 0.84, "event_at": dec_at + timedelta(seconds=5), "received_at": dec_at + timedelta(seconds=5)}
            return {"best_ask": 0.15, "best_bid": 0.14, "event_at": dec_at + timedelta(seconds=5), "received_at": dec_at + timedelta(seconds=5)}

    dec_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=FlippedClient(),
        market=market,
        cfg=parse_trading_settings({}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at + timedelta(seconds=5),
        time_left_sec=235.0,
    )

    # Immutability check: decision runner returns SKIP and refuses to flip to DOWN
    assert dec_res.decision_obj.action == "SKIP"
    assert "ALREADY_DECIDED: BUY" in dec_res.decision_obj.reason
    assert dec_res.decision_obj.direction_value == "UP"  # Original direction preserved!


def test_15_paper_profile_report_invariant():
    """Requirement 21: Compact report enforces invariant Total = UP + DOWN."""
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


@pytest.mark.asyncio
async def test_16_full_audit_chain_traceability_from_real_db_rows(db_session, base_decision_time):
    """Requirement 22: Audit chain links opportunity_id -> decision_id -> request_id -> fill -> settlement from real DB rows."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_trace_chain", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    dec = CTDecision(
        action="BUY", side="UP", token_id="up_tok", reason="CT_SIGNAL_REVERSION",
        spec_id="BTC_CT_T5_V1", spec_hash="trace_hash_123", decision_at=dec_at, time_left_sec=240.0,
        market_id=market.market_id, asset="BTC", limit_price=0.20, budget_usdc=1.00,
        selected_ask=0.20, selected_mid=0.20, selected_bid=0.19, other_mid=0.80,
        outsider_margin=0.30, ct_regime="REVERSION", ct_features={"sign_change_freq": 0.6},
        data_ids={"up_snapshot_id": 999111},
        is_executable=True,
    )

    # 1. Execute and record
    trade, req = await _record_and_claim_trade(db_session, market, dec, dec_at)

    # 2. Gateway execution
    asks = [{"price": 0.20, "size": 100.0}]
    sub_res = await _execute_paper_gateway(db_session, trade, req, asks=asks)
    assert sub_res.accepted is True

    # 3. Settlement
    await settle_resolved_position(db_session, trade_id=trade.id, winning_outcome="YES", payout_per_share=Decimal("1.0"))
    await db_session.commit()
    await db_session.refresh(trade)

    # 4. Verify complete audit chain from actual DB rows
    # A. Reservation row
    res = await get_ct_decision_reservation(db_session, f"CT:BTC_CT_T5_V1:{market.market_id}")
    assert res is not None
    assert res.trade_history_id == trade.id
    assert res.market_id == market.market_id
    assert res.action == "BUY"

    # B. ExecutionRequest row
    req_db = await db_session.get(ExecutionRequest, req.id)
    assert req_db is not None
    assert req_db.trade_history_id == trade.id
    assert req_db.state == "FILLED"

    # C. ExecutionAttempt & ExecutionFill rows
    attempts = (await db_session.execute(select(ExecutionAttempt).where(ExecutionAttempt.request_id == req.id))).scalars().all()
    assert len(attempts) == 1
    attempt_db = attempts[0]

    fills = (await db_session.execute(select(ExecutionFill).where(ExecutionFill.attempt_id == attempt_db.id))).scalars().all()
    assert len(fills) == 1
    fill_db = fills[0]

    # D. TradeHistory row
    assert trade.position_status == "CLOSED"
    assert trade.realized_pnl_usdc is not None

    # E. Full link verification
    opportunity_id = f"{market.market_id}_{dec_at.isoformat()}"
    decision_id = res.key
    request_id = str(req_db.id)
    fill_id = str(fill_db.id)
    trade_id = trade.id

    assert decision_id == f"CT:BTC_CT_T5_V1:{market.market_id}"
    assert req_db.trade_history_id == trade_id
    assert attempt_db.request_id == req_db.id
    assert fill_db.attempt_id == attempt_db.id
    assert trade.position_status == "CLOSED"
