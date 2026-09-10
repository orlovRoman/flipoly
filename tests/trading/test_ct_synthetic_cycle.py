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

import asyncio
import math
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from polyflip.db.models import LiveMarket, TradeHistory, CTDecisionReservation, MarketSnapshot, OrderbookDepthSnapshot
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
from polyflip.trading.pre_trade_validator import validate_pre_trade
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.trade_recorder import execute_and_record, EnqueueRejected
from polyflip.trading.decision_runners import decide_ct_outsider_mode, DecisionResult
from polyflip.trading.ct_reservation import (
    reserve_ct_decision,
    get_ct_decision_reservation,
)
from polyflip.execution.worker import (
    claim_one,
    rebuild_trade_accounting,
    process_ready_requests,
)
from polyflip.execution.settlement_service import settle_resolved_position
from polyflip.execution.gateways.fake import FakeExecutionGateway
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


async def _seed_market_snapshots(
    db_session,
    market_id: str,
    base_time: datetime,
    up_prices: list[float],
    down_prices: list[float] | None = None,
    asset: str = "BTC",
):
    for i, p in enumerate(up_prices):
        t = base_time - timedelta(minutes=len(up_prices) - 1 - i)
        snap = MarketSnapshot(
            market_id=market_id,
            asset=asset,
            recorded_at=t,
            market_timestamp=t,
            received_timestamp=t,
            mid_price=p,
            poly_up_mid=p,
            poly_down_mid=down_prices[i] if down_prices else (1.0 - p),
            time_left_min=float(len(up_prices) - 1 - i),
            volume_5min=100.0,
            price_velocity=0.0,
            hour_of_day=t.hour,
            final_outcome="PENDING",
        )
        db_session.add(snap)
    await db_session.flush()


async def _run_production_paper_cycle(
    db_session,
    market: LiveMarket,
    api_client: Any,
    start_time: datetime,
    *,
    gateway: Any = None,
    quote_provider: Any = None,
    fee_rate: Decimal = Decimal("0.002"),
    models_cache: Any = None,
    crypto_predictor: Any = None,
    raw_settings: dict | None = None,
    decision_at: datetime | None = None,
) -> tuple[TradeHistory | None, DecisionResult | None]:
    import polyflip.execution.worker as worker_module

    cfg = parse_trading_settings(raw_settings or {"TRADING_MODE": "ct_outsider"})
    eff_dec_at = decision_at if decision_at is not None else start_time
    time_left_sec = (market.end_time_est - eff_dec_at).total_seconds()

    decision_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=api_client,
        market=market,
        cfg=cfg,
        raw_settings=raw_settings or {},
        models_cache=models_cache,
        crypto_predictor=crypto_predictor,
        start_time=start_time,
        time_left_sec=time_left_sec,
        execution_mode="PAPER",
        decision_at=eff_dec_at,
    )

    if not decision_res or not decision_res.decision_obj or decision_res.decision_obj.action == "SKIP":
        return None, decision_res

    validation = await validate_pre_trade(
        db_session=db_session,
        api_client=api_client,
        market=market,
        decision_obj=decision_res.decision_obj,
        cfg=cfg,
        asset_mode="FAVORITE",
        asset_min_edge=0.0,
        asset_max_price=0.95,
        p_flip=decision_res.p_flip,
        model_ver=decision_res.model_ver,
    )
    if not validation.valid:
        return None, decision_res

    await execute_and_record(
        db_session=db_session,
        market=market,
        decision_obj=decision_res.decision_obj,
        validation=validation,
        asset_mode="FAVORITE",
        active_features="ct_features",
        p_flip=decision_res.p_flip,
        model_ver=decision_res.model_ver,
        cfg=cfg,
        existing_skipped=None,
        start_time=start_time,
        decision_at=getattr(decision_res, "decision_at", None),
    )
    await db_session.commit()

    if gateway is None and quote_provider is not None:
        gateway = FakeExecutionGateway(
            profile="LIVE_PARITY",
            quote_provider=quote_provider,
            fee_rate=fee_rate,
            slippage_pct=Decimal("0"),
            delay_sec=0.0,
        )

    bind = getattr(db_session, "bind", None) or (db_session.get_bind() if hasattr(db_session, "get_bind") else None)
    session_factory = async_sessionmaker(bind, expire_on_commit=False)
    orig_worker_session = worker_module.async_session
    worker_module.async_session = session_factory
    try:
        await process_ready_requests(gateway=gateway, quote_provider=quote_provider)
    finally:
        worker_module.async_session = orig_worker_session

    trade_stmt = select(TradeHistory).where(TradeHistory.market_id == market.market_id)
    trade = (await db_session.execute(trade_stmt)).scalars().first()
    if trade is not None:
        await db_session.refresh(trade)
    return trade, decision_res


# ==============================================================================
# Requirements 10-15: Real PAPER Execution Pipeline Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_01_synthetic_up_outsider_buy_and_full_fill_win(db_session, base_decision_time):
    """Requirement 10, 11: UP outsider BUY executes through production paper cycle with full fill and winning settlement."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_up_win", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    # Seed 8 historical snapshots (REVERSION regime: alternating prices)
    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23],
    )

    class CausalClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=2),
                "received_at": dec_at - timedelta(seconds=2),
            }

    async def quote_provider(_tok: str):
        return {
            "asks": [{"price": 0.20, "size": 100.0}],
            "bids": [{"price": 0.19, "size": 100.0}],
            "best_ask": 0.20,
            "best_bid": 0.19,
        }

    # Run full production cycle: dispatcher -> validator -> recorder -> worker.process_ready_requests
    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        CausalClient(),
        dec_at,
        quote_provider=quote_provider,
        fee_rate=Decimal("0.002"),
    )

    assert dec_res is not None
    assert dec_res.decision_obj.action == "BUY_YES"
    assert dec_res.decision_obj.direction_value == "UP"
    assert dec_res.decision_obj.buy_price == 0.20

    assert trade is not None
    assert trade.position_status == "OPEN"
    # Budget 1.00 USDC, ask 0.20, fee 0.002 -> shares = 1.00 / (0.20 * 1.002) = 4.99002
    assert math.isclose(float(trade.entry_filled_shares), 4.99002, abs_tol=1e-4)
    # Total spend is capped exactly at 1.00 USDC (0.998004 gross + 0.001996 fee)
    assert math.isclose(float(trade.entry_cost_usdc), 1.00000, abs_tol=1e-4)

    # Settlement: UP (YES) wins
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
    # PnL = 4.99002 * 1.0 - 1.00000 = +3.99002 USDC
    assert math.isclose(float(trade.realized_pnl_usdc), 3.99002, abs_tol=1e-4)


@pytest.mark.asyncio
async def test_02_synthetic_down_outsider_buy_and_full_fill_win(db_session, base_decision_time):
    """Requirement 10, 11: DOWN outsider BUY executes through production paper cycle with full fill and winning settlement."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_down_win", end_time=dec_at + timedelta(seconds=250))
    await db_session.commit()

    # Seed 8 snapshots for DOWN outsider reversion (DOWN mid alternating around 0.25)
    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.78, 0.72, 0.79, 0.73, 0.77, 0.71, 0.78, 0.72],
        down_prices=[0.22, 0.28, 0.21, 0.27, 0.23, 0.29, 0.22, 0.28],
    )

    class DownClient:
        async def get_market_prices(self, tok, **kwargs):
            if tok == market.no_token_id or tok == "down_tok":
                return {
                    "best_ask": 0.25,
                    "best_bid": 0.24,
                    "current_no_price": 0.25,
                    "event_at": dec_at - timedelta(seconds=1),
                    "received_at": dec_at - timedelta(seconds=1),
                }
            return {
                "best_ask": 0.76,
                "best_bid": 0.74,
                "best_ask_no": 0.25,
                "best_bid_no": 0.24,
                "current_yes_price": 0.75,
                "current_no_price": 0.25,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    async def quote_provider(_tok: str):
        return {
            "asks": [{"price": 0.25, "size": 50.0}],
            "bids": [{"price": 0.24, "size": 50.0}],
            "best_ask": 0.25,
            "best_bid": 0.24,
        }

    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        DownClient(),
        dec_at,
        quote_provider=quote_provider,
        fee_rate=Decimal("0.002"),
    )

    assert dec_res is not None
    assert dec_res.decision_obj.action == "BUY_NO"
    assert dec_res.decision_obj.direction_value == "DOWN"
    assert dec_res.decision_obj.buy_price == 0.25

    assert trade is not None
    assert trade.position_status == "OPEN"
    # Budget 1.00 USDC, ask 0.25, fee 0.002 -> shares = 1.00 / (0.25 * 1.002) = 3.99202
    assert math.isclose(float(trade.entry_filled_shares), 3.99202, abs_tol=1e-4)
    assert math.isclose(float(trade.entry_cost_usdc), 1.00000, abs_tol=1e-4)

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
    # PnL = 3.99202 * 1.0 - 1.00000 = +2.99202 USDC
    assert math.isclose(float(trade.realized_pnl_usdc), 2.99202, abs_tol=1e-4)


@pytest.mark.asyncio
async def test_02b_down_history_falls_back_to_orderbook_depth(
    db_session, base_decision_time
):
    # Regression: live collector already stores both token books in
    # OrderbookDepthSnapshot, while legacy MarketSnapshot rows may have only
    # the YES alias populated. DOWN must not be reported as missing history.
    dec_at = base_decision_time
    market = _make_live_market(
        db_session,
        market_id="mkt_down_depth_fallback",
        end_time=dec_at + timedelta(seconds=250),
    )
    await db_session.commit()

    up_prices = [0.78, 0.72, 0.79, 0.73, 0.77, 0.71, 0.78, 0.72]
    down_prices = [0.22, 0.28, 0.21, 0.27, 0.23, 0.29, 0.22, 0.28]
    for i, up_price in enumerate(up_prices):
        t = dec_at - timedelta(minutes=len(up_prices) - 1 - i)
        db_session.add(
            MarketSnapshot(
                market_id=market.market_id,
                asset="BTC",
                recorded_at=t,
                market_timestamp=t,
                received_timestamp=t,
                mid_price=up_price,
                poly_up_mid=up_price,
                poly_down_mid=None,
                time_left_min=4.0,
                volume_5min=100.0,
                price_velocity=0.0,
                hour_of_day=t.hour,
                final_outcome="PENDING",
            )
        )
        down_price = down_prices[i]
        db_session.add(
            OrderbookDepthSnapshot(
                market_id=market.market_id,
                token_id=market.no_token_id,
                outcome_side="NO",
                event_at=t,
                received_at=t,
                bids=[{"price": down_price - 0.005, "size": 50.0}],
                asks=[{"price": down_price + 0.005, "size": 50.0}],
                quality_status="VALID",
                best_bid_price=down_price - 0.005,
                best_ask_price=down_price + 0.005,
                best_bid_size=50.0,
                best_ask_size=50.0,
            )
        )
    await db_session.commit()

    class DepthFallbackClient:
        async def get_market_prices(self, tok, **kwargs):
            if tok == market.no_token_id:
                return {
                    "best_ask": 0.25,
                    "best_bid": 0.24,
                    "current_no_price": 0.25,
                    "event_at": dec_at - timedelta(seconds=1),
                    "received_at": dec_at - timedelta(seconds=1),
                }
            return {
                "best_ask": 0.76,
                "best_bid": 0.74,
                "best_ask_no": 0.25,
                "best_bid_no": 0.24,
                "current_yes_price": 0.75,
                "current_no_price": 0.25,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    async def quote_provider(_tok: str):
        return {
            "asks": [{"price": 0.25, "size": 50.0}],
            "bids": [{"price": 0.24, "size": 50.0}],
            "best_ask": 0.25,
            "best_bid": 0.24,
        }

    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        DepthFallbackClient(),
        dec_at,
        quote_provider=quote_provider,
        fee_rate=Decimal("0.002"),
    )

    assert dec_res is not None
    assert dec_res.decision_obj is not None
    assert dec_res.decision_obj.action == "BUY_NO"
    assert dec_res.decision_obj.direction_value == "DOWN"
    assert trade is not None


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

    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.05, 0.06, 0.05, 0.06, 0.05, 0.06, 0.05, 0.06],
    )

    class PartialClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.05,
                "best_bid": 0.04,
                "best_ask_no": 0.95,
                "best_bid_no": 0.94,
                "current_yes_price": 0.05,
                "current_no_price": 0.95,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    # Orderbook only has 10 shares @ 0.05 ($0.50 depth) for $1.00 budget
    async def quote_provider(_tok: str):
        return {
            "asks": [{"price": 0.05, "size": 10.0}],
            "bids": [{"price": 0.04, "size": 10.0}],
            "best_ask": 0.05,
            "best_bid": 0.04,
        }

    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        PartialClient(),
        dec_at,
        quote_provider=quote_provider,
        fee_rate=Decimal("0.002"),
    )

    assert trade is not None
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
    # Realized loss is strictly -0.501 USDC, NOT -1.00 USDC! Unspent $0.499 is preserved.
    assert math.isclose(float(trade.realized_pnl_usdc), -0.501, abs_tol=1e-5)
    assert trade.realized_pnl_usdc > Decimal("-1.00")


@pytest.mark.asyncio
async def test_08_synthetic_limit_price_change_rejection(db_session, base_decision_time):
    """Requirement 13: If orderbook ask moves above limit price, gateway rejects order with 0 fills."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_price_moved", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23],
    )

    class DecisionClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    # At execution, orderbook ask moved up to 0.21 (worse than limit_price 0.20)
    async def moved_quote_provider(_tok: str):
        return {
            "asks": [{"price": 0.21, "size": 100.0}],
            "bids": [{"price": 0.20, "size": 100.0}],
            "best_ask": 0.21,
            "best_bid": 0.20,
        }

    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        DecisionClient(),
        dec_at,
        quote_provider=moved_quote_provider,
    )

    assert trade is not None
    assert trade.entry_filled_shares == Decimal("0")
    assert trade.position_status == "ENTRY_FAILED"
    req = (await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id))).scalar_one()
    assert req.state == "REJECTED"


@pytest.mark.asyncio
async def test_09_synthetic_multi_level_orderbook_vwap(db_session, base_decision_time):
    """Requirement 14: Multi-level book consumes multiple levels and calculates correct VWAP entry price without overrides."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_vwap", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12],
    )

    class VwapClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.12,
                "best_bid": 0.11,
                "best_ask_no": 0.88,
                "best_bid_no": 0.87,
                "current_yes_price": 0.12,
                "current_no_price": 0.88,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    # 2 ask levels:
    # Level 1: 5 shares @ 0.10 = $0.50
    # Level 2: 10 shares @ 0.12 available
    # Budget: $1.00, limit: 0.12
    # In outbox: requested_shares = 1.00 / 0.12 = 8.333333 shares
    # Level 1 fills 5.0 shares @ 0.10 ($0.50)
    # Level 2 fills 3.333333 shares @ 0.12 ($0.40)
    # Total shares: 8.333333, Total gross: 0.90 USDC
    # VWAP = 0.90 / 8.333333 = 0.108 USDC/share
    async def multi_level_provider(_tok: str):
        return {
            "asks": [
                {"price": 0.10, "size": 5.0},
                {"price": 0.12, "size": 10.0},
            ],
            "bids": [{"price": 0.09, "size": 10.0}],
            "best_ask": 0.10,
            "best_bid": 0.09,
        }

    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        VwapClient(),
        dec_at,
        quote_provider=multi_level_provider,
        fee_rate=Decimal("0"),
    )

    assert trade is not None
    assert trade.position_status == "OPEN"
    assert math.isclose(float(trade.entry_filled_shares), 8.333333, abs_tol=1e-5)
    assert math.isclose(float(trade.entry_cost_usdc), 0.90, abs_tol=1e-5)
    assert math.isclose(float(trade.executed_price), 0.108, abs_tol=1e-5)


@pytest.mark.asyncio
async def test_10_synthetic_partial_fill_settlement_win_and_loss(db_session, base_decision_time):
    """Requirement 15: Partial fill settlement: WIN (+9.499 USDC), LOSS (-0.501 USDC), and repeat settlement no-op."""
    dec_at = base_decision_time

    # --- Part A: WIN ---
    market_win = _make_live_market(db_session, market_id="mkt_part_win", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()
    await _seed_market_snapshots(db_session, market_win.market_id, dec_at, [0.05, 0.06, 0.05, 0.06, 0.05, 0.06, 0.05, 0.06])

    class PartClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.05,
                "best_bid": 0.04,
                "best_ask_no": 0.95,
                "best_bid_no": 0.94,
                "current_yes_price": 0.05,
                "current_no_price": 0.95,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    async def part_provider(_tok: str):
        return {
            "asks": [{"price": 0.05, "size": 10.0}],
            "bids": [{"price": 0.04, "size": 10.0}],
            "best_ask": 0.05,
            "best_bid": 0.04,
        }

    trade_win, _ = await _run_production_paper_cycle(
        db_session, market_win, PartClient(), dec_at, quote_provider=part_provider, fee_rate=Decimal("0.002")
    )

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
    await _seed_market_snapshots(db_session, market_loss.market_id, dec_at, [0.05, 0.06, 0.05, 0.06, 0.05, 0.06, 0.05, 0.06])

    trade_loss, _ = await _run_production_paper_cycle(
        db_session, market_loss, PartClient(), dec_at, quote_provider=part_provider, fee_rate=Decimal("0.002")
    )

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
async def test_11_concurrency_two_sessions_atomic_reservation(tmp_path, base_decision_time):
    """Requirement 16, 17: Concurrent decision execution across two independent sessions with barrier yields exactly 1 trade and 1 rejected duplicate."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from polyflip.db.models import Base

    dec_at = base_decision_time
    db_file = tmp_path / "concur_test.db"
    concur_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    async with concur_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(concur_engine, expire_on_commit=False)

    try:
        async with session_maker() as init_session:
            market = _make_live_market(init_session, market_id="mkt_concur", end_time=dec_at + timedelta(seconds=240))
            await init_session.commit()
            await _seed_market_snapshots(
                init_session,
                market.market_id,
                dec_at,
                [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23],
            )
            await init_session.commit()

        class ConcurClient:
            async def get_market_prices(self, tok, **kwargs):
                return {
                    "best_ask": 0.20,
                    "best_bid": 0.19,
                    "best_ask_no": 0.80,
                    "best_bid_no": 0.79,
                    "current_yes_price": 0.20,
                    "current_no_price": 0.80,
                    "event_at": dec_at - timedelta(seconds=2),
                    "received_at": dec_at - timedelta(seconds=2),
                }

        barrier = asyncio.Barrier(2)
        results: list[tuple[str, Any, Exception | None]] = []

        async def attempt_record(worker_id: str):
            async with session_maker() as session:
                mkt = (await session.execute(select(LiveMarket).where(LiveMarket.market_id == "mkt_concur"))).scalar_one()
                cfg = parse_trading_settings({"TRADING_MODE": "ct_outsider"})

                # Production dispatcher
                dec_res = await decide_ct_outsider_mode(
                    db_session=session,
                    api_client=ConcurClient(),
                    market=mkt,
                    cfg=cfg,
                    raw_settings={"TRADING_MODE": "ct_outsider"},
                    models_cache=None,
                    crypto_predictor=None,
                    start_time=dec_at,
                    time_left_sec=240.0,
                    execution_mode="PAPER",
                    decision_at=dec_at,
                )
                assert dec_res is not None
                assert dec_res.decision_obj is not None
                assert dec_res.decision_obj.action == "BUY_YES"

                # Production validator
                val = await validate_pre_trade(
                    db_session=session,
                    api_client=ConcurClient(),
                    market=mkt,
                    decision_obj=dec_res.decision_obj,
                    cfg=cfg,
                    asset_mode="FAVORITE",
                    asset_min_edge=0.0,
                    asset_max_price=0.95,
                    p_flip=dec_res.p_flip,
                    model_ver=dec_res.model_ver,
                )
                assert val.valid is True

                # Synchronize right before atomic reservation and enqueue
                await barrier.wait()
                try:
                    await execute_and_record(
                        db_session=session,
                        market=mkt,
                        decision_obj=dec_res.decision_obj,
                        validation=val,
                        asset_mode="FAVORITE",
                        active_features="ct_features",
                        p_flip=dec_res.p_flip,
                        model_ver=dec_res.model_ver,
                        cfg=cfg,
                        existing_skipped=None,
                        start_time=dec_at,
                        decision_at=dec_res.decision_at,
                    )
                    await session.commit()
                    results.append((worker_id, "COMMITTED", None))
                except EnqueueRejected as exc:
                    await session.commit()
                    results.append((worker_id, None, exc))
                except Exception as exc:
                    await session.rollback()
                    results.append((worker_id, None, exc))

        await asyncio.gather(attempt_record("worker_1"), attempt_record("worker_2"))

        # Exactly one worker succeeded and one raised EnqueueRejected
        succeeded = [r for r in results if r[1] is not None]
        failed = [r for r in results if r[2] is not None]
        assert len(succeeded) == 1
        assert len(failed) == 1
        assert isinstance(failed[0][2], EnqueueRejected)
        assert "ActiveExecutionConflict" in str(failed[0][2])
        assert "already reserved" in str(failed[0][2])

        # Clean verification session: exactly 1 trade, 1 reservation (repeat_count=1), and 1 READY request
        async with session_maker() as verify_session:
            trades = (await verify_session.execute(select(TradeHistory).where(TradeHistory.market_id == "mkt_concur"))).scalars().all()
            assert len(trades) == 1
            res = await get_ct_decision_reservation(verify_session, "CT:BTC_CT_T5_V1:mkt_concur")
            assert res is not None
            assert res.repeat_count == 1
            requests = (await verify_session.execute(select(ExecutionRequest).where(ExecutionRequest.market_id == "mkt_concur"))).scalars().all()
            assert len(requests) == 1
            assert requests[0].state == "READY"

        # Production worker processes the single winning READY request to completion
        import polyflip.execution.worker as worker_module
        orig_worker_session = worker_module.async_session
        worker_module.async_session = session_maker
        try:
            async def quote_prov(_tok: str):
                return {"asks": [{"price": 0.20, "size": 50.0}], "bids": [{"price": 0.19, "size": 50.0}], "best_ask": 0.20, "best_bid": 0.19}
            gateway = FakeExecutionGateway(profile="LIVE_PARITY", quote_provider=quote_prov, fee_rate=Decimal("0.002"))
            await process_ready_requests(gateway=gateway, quote_provider=quote_prov)
        finally:
            worker_module.async_session = orig_worker_session

        async with session_maker() as final_session:
            req_final = (await final_session.execute(select(ExecutionRequest).where(ExecutionRequest.market_id == "mkt_concur"))).scalar_one()
            assert req_final.state in {"FILLED", "PARTIALLY_FILLED_FINAL"}
            trade_final = (await final_session.execute(select(TradeHistory).where(TradeHistory.market_id == "mkt_concur"))).scalar_one()
            assert trade_final.position_status == "OPEN"
            assert math.isclose(float(trade_final.entry_filled_shares), 4.99002, abs_tol=1e-4)
    finally:
        await concur_engine.dispose()


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_11_postgres_concurrency_two_sessions_atomic_reservation(pg_session_factory, base_decision_time):
    """Requirement 13, 14: Real PostgreSQL concurrency verification with two independent sessions and barrier."""
    dec_at = base_decision_time
    mkt_id = f"mkt_pg_concur_{uuid.uuid4().hex[:8]}"

    async with pg_session_factory() as init_session:
        market = _make_live_market(init_session, market_id=mkt_id, end_time=dec_at + timedelta(seconds=240))
        await init_session.commit()
        await _seed_market_snapshots(
            init_session,
            mkt_id,
            dec_at,
            [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23],
        )
        await init_session.commit()

    class ConcurClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=2),
                "received_at": dec_at - timedelta(seconds=2),
            }

    barrier = asyncio.Barrier(2)
    results: list[tuple[str, Any, Exception | None]] = []

    async def attempt_record(worker_id: str):
        async with pg_session_factory() as session:
            mkt = (await session.execute(select(LiveMarket).where(LiveMarket.market_id == mkt_id))).scalar_one()
            cfg = parse_trading_settings({"TRADING_MODE": "ct_outsider"})

            dec_res = await decide_ct_outsider_mode(
                db_session=session,
                api_client=ConcurClient(),
                market=mkt,
                cfg=cfg,
                raw_settings={"TRADING_MODE": "ct_outsider"},
                models_cache=None,
                crypto_predictor=None,
                start_time=dec_at,
                time_left_sec=240.0,
                execution_mode="PAPER",
                decision_at=dec_at,
            )
            assert dec_res is not None
            assert dec_res.decision_obj.action == "BUY_YES"

            val = await validate_pre_trade(
                db_session=session,
                api_client=ConcurClient(),
                market=mkt,
                decision_obj=dec_res.decision_obj,
                cfg=cfg,
                asset_mode="FAVORITE",
                asset_min_edge=0.0,
                asset_max_price=0.95,
                p_flip=dec_res.p_flip,
                model_ver=dec_res.model_ver,
            )
            assert val.valid is True

            await barrier.wait()
            try:
                await execute_and_record(
                    db_session=session,
                    market=mkt,
                    decision_obj=dec_res.decision_obj,
                    validation=val,
                    asset_mode="FAVORITE",
                    active_features="ct_features",
                    p_flip=dec_res.p_flip,
                    model_ver=dec_res.model_ver,
                    cfg=cfg,
                    existing_skipped=None,
                    start_time=dec_at,
                    decision_at=dec_res.decision_at,
                )
                await session.commit()
                results.append((worker_id, "COMMITTED", None))
            except EnqueueRejected as exc:
                await session.commit()
                results.append((worker_id, None, exc))
            except Exception as exc:
                await session.rollback()
                results.append((worker_id, None, exc))

    await asyncio.gather(attempt_record("pg_worker_1"), attempt_record("pg_worker_2"))

    succeeded = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[2] is not None]
    assert len(succeeded) == 1
    assert len(failed) == 1
    assert isinstance(failed[0][2], EnqueueRejected)

    async with pg_session_factory() as verify_session:
        trades = (await verify_session.execute(select(TradeHistory).where(TradeHistory.market_id == mkt_id))).scalars().all()
        assert len(trades) == 1
        res = await get_ct_decision_reservation(verify_session, f"CT:BTC_CT_T5_V1:{mkt_id}")
        assert res is not None
        assert res.repeat_count == 1
        requests = (await verify_session.execute(select(ExecutionRequest).where(ExecutionRequest.market_id == mkt_id))).scalars().all()
        assert len(requests) == 1
        assert requests[0].state == "READY"


@pytest.mark.asyncio
async def test_11b_concurrency_rollback_releases_lock(engine, base_decision_time):
    """Requirement 17: If session 1 reserves but rolls back, session 2 acquires lock and commits cleanly with repeat_count=0."""
    dec_at = base_decision_time
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with session_maker() as init_session:
        _make_live_market(init_session, market_id="mkt_concur_rb", end_time=dec_at + timedelta(seconds=240))
        await init_session.commit()

    # Session 1 reserves but rolls back
    async with session_maker() as s1:
        is_first1, res1 = await reserve_ct_decision(
            s1,
            key="CT:BTC_CT_T5_V1:mkt_concur_rb",
            market_id="mkt_concur_rb",
            spec_id="BTC_CT_T5_V1",
            action="BUY",
            decision_at=dec_at,
            side="UP",
            limit_price=0.20,
            budget_usdc=1.00,
            reason="CT_SIGNAL_REVERSION",
        )
        assert is_first1 is True
        await s1.rollback()

    # Session 2 attempts reservation and commits
    async with session_maker() as s2:
        is_first2, res2 = await reserve_ct_decision(
            s2,
            key="CT:BTC_CT_T5_V1:mkt_concur_rb",
            market_id="mkt_concur_rb",
            spec_id="BTC_CT_T5_V1",
            action="BUY",
            decision_at=dec_at,
            side="UP",
            limit_price=0.20,
            budget_usdc=1.00,
            reason="CT_SIGNAL_REVERSION",
        )
        await s2.commit()
        assert is_first2 is True
        assert res2.repeat_count == 0

    async with session_maker() as verify_session:
        res_db = await get_ct_decision_reservation(verify_session, "CT:BTC_CT_T5_V1:mkt_concur_rb")
        assert res_db is not None
        assert res_db.repeat_count == 0
        assert res_db.action == "BUY"


@pytest.mark.asyncio
async def test_11c_atomic_repeat_count_increment_under_concurrency(engine, base_decision_time):
    """Requirement 17: 3 concurrent repeat attempts with barrier atomically increment repeat_count to 3 without lost updates."""
    dec_at = base_decision_time
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with session_maker() as init_session:
        _make_live_market(init_session, market_id="mkt_concur_rep", end_time=dec_at + timedelta(seconds=240))
        # Initial committed reservation
        is_first, res = await reserve_ct_decision(
            init_session,
            key="CT:BTC_CT_T5_V1:mkt_concur_rep",
            market_id="mkt_concur_rep",
            spec_id="BTC_CT_T5_V1",
            action="BUY",
            decision_at=dec_at,
            side="UP",
            limit_price=0.20,
            budget_usdc=1.00,
            reason="CT_SIGNAL_REVERSION",
        )
        await init_session.commit()
        assert is_first is True
        assert res.repeat_count == 0

    barrier = asyncio.Barrier(3)
    results: list[tuple[int, bool, int]] = []

    async def repeat_worker(worker_id: int):
        async with session_maker() as session:
            await barrier.wait()
            is_first_w, res_w = await reserve_ct_decision(
                session,
                key="CT:BTC_CT_T5_V1:mkt_concur_rep",
                market_id="mkt_concur_rep",
                spec_id="BTC_CT_T5_V1",
                action="BUY",
                decision_at=dec_at,
            )
            await session.commit()
            results.append((worker_id, is_first_w, res_w.repeat_count))

    await asyncio.gather(repeat_worker(1), repeat_worker(2), repeat_worker(3))

    assert len(results) == 3
    assert all(r[1] is False for r in results)

    async with session_maker() as verify_session:
        final_res = await get_ct_decision_reservation(verify_session, "CT:BTC_CT_T5_V1:mkt_concur_rep")
        assert final_res is not None
        assert final_res.repeat_count == 3
        assert final_res.action == "BUY"
        assert final_res.side == "UP"


@pytest.mark.asyncio
async def test_11d_repeat_count_identity_map_refresh(db_session, base_decision_time):
    """Requirement 17: reserve_ct_decision refreshes existing instance if already cached in session identity map."""
    dec_at = base_decision_time
    _make_live_market(db_session, market_id="mkt_id_map", end_time=dec_at + timedelta(seconds=240))
    # Initial reservation
    is_first, res = await reserve_ct_decision(
        db_session,
        key="CT:BTC_CT_T5_V1:mkt_id_map",
        market_id="mkt_id_map",
        spec_id="BTC_CT_T5_V1",
        action="BUY",
        decision_at=dec_at,
        side="UP",
        limit_price=0.20,
        budget_usdc=1.00,
        reason="CT_SIGNAL_REVERSION",
    )
    await db_session.commit()
    assert is_first is True
    assert res.repeat_count == 0

    # Load into identity map
    cached_res = await get_ct_decision_reservation(db_session, "CT:BTC_CT_T5_V1:mkt_id_map")
    assert cached_res is not None
    assert cached_res.repeat_count == 0

    # Repeat reservation in same session must refresh cached entity
    is_first2, res2 = await reserve_ct_decision(
        db_session,
        key="CT:BTC_CT_T5_V1:mkt_id_map",
        market_id="mkt_id_map",
        spec_id="BTC_CT_T5_V1",
        action="BUY",
        decision_at=dec_at,
    )
    await db_session.commit()
    assert is_first2 is False
    assert res2.repeat_count == 1
    assert cached_res.repeat_count == 1


@pytest.mark.asyncio
async def test_12_rerun_after_settlement_blocked(db_session, base_decision_time):
    """Requirement 18: Re-running decision on the same market after settlement returns ALREADY_DECIDED."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_rerun", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()

    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23],
    )

    class RerunClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    async def quote_prov(_tok: str):
        return {"asks": [{"price": 0.20, "size": 50.0}], "bids": [{"price": 0.19, "size": 50.0}], "best_ask": 0.20, "best_bid": 0.19}

    # Initial cycle completes through production paper cycle and settles
    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        RerunClient(),
        dec_at,
        quote_provider=quote_prov,
        fee_rate=Decimal("0.002"),
    )
    assert trade is not None
    await settle_resolved_position(db_session, trade_id=trade.id, winning_outcome="YES", payout_per_share=Decimal("1.0"))
    await db_session.commit()
    await db_session.refresh(trade)
    assert trade.position_status == "CLOSED"

    # Subsequent cycle polls the same market via decide_ct_outsider_mode
    dec_res2 = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=RerunClient(),
        market=market,
        cfg=parse_trading_settings({}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at + timedelta(seconds=10),
        time_left_sec=230.0,
        execution_mode="PAPER",
    )

    assert dec_res2.decision_obj.action == "SKIP"
    assert "ALREADY_DECIDED: BUY" in dec_res2.decision_obj.reason
    assert dec_res2.decision_obj.decision_details.get("repeat") is True

    # Ensure no new trade was created
    trades = (await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == market.market_id))).scalars().all()
    assert len(trades) == 1
    assert trades[0].position_status == "CLOSED"


@pytest.mark.asyncio
async def test_13_recovery_at_three_failure_points(db_session, base_decision_time):
    """Requirement 19: Interruption & recovery at 3 failure points: after reservation, after enqueue, after fill before ack."""
    from unittest.mock import patch
    dec_at = base_decision_time

    # Point 1: Crash between decision reservation and outbox enqueue
    # When execute_and_record fails during enqueue, the nested savepoint rolls back
    # so NO lone BUY reservation is left in the DB, and a retry succeeds.
    mkt1 = _make_live_market(db_session, market_id="mkt_crash_1", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()
    await _seed_market_snapshots(db_session, mkt1.market_id, dec_at, [0.20, 0.22, 0.20, 0.22, 0.20, 0.22, 0.20, 0.22])

    class CrashClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    # Inject failure during enqueue
    with patch("polyflip.trading.trade_recorder.enqueue_open_request", side_effect=RuntimeError("Simulated network/DB failure")):
        with pytest.raises(RuntimeError, match="Simulated network/DB failure"):
            await _run_production_paper_cycle(db_session, mkt1, CrashClient(), dec_at)

    # Verify rollback left NO lone BUY reservation or orphaned trade
    res1 = await get_ct_decision_reservation(db_session, f"CT:BTC_CT_T5_V1:{mkt1.market_id}")
    assert res1 is None
    trades1 = (await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == mkt1.market_id))).scalars().all()
    assert len(trades1) == 0

    # Retry succeeds cleanly!
    async def retry_quote_prov(_tok: str):
        return {"asks": [{"price": 0.20, "size": 50.0}], "bids": [{"price": 0.19, "size": 50.0}], "best_ask": 0.20, "best_bid": 0.19}
    trade1_recovered, _ = await _run_production_paper_cycle(db_session, mkt1, CrashClient(), dec_at, quote_provider=retry_quote_prov)
    assert trade1_recovered is not None
    assert trade1_recovered.position_status == "OPEN"

    # Point 2: Crash after enqueue (ExecutionRequest in state READY in outbox)
    # The process that called execute_and_record crashed after enqueuing.
    # Fresh worker process runs process_ready_requests and discovers it from DB without pre-selected objects.
    mkt2 = _make_live_market(db_session, market_id="mkt_crash_2", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()
    await _seed_market_snapshots(db_session, mkt2.market_id, dec_at, [0.20, 0.22, 0.20, 0.22, 0.20, 0.22, 0.20, 0.22])

    dec2_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=CrashClient(),
        market=mkt2,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at,
        time_left_sec=240.0,
        execution_mode="PAPER",
        decision_at=dec_at,
    )
    val2 = await validate_pre_trade(
        db_session=db_session,
        api_client=CrashClient(),
        market=mkt2,
        decision_obj=dec2_res.decision_obj,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        asset_mode="FAVORITE",
        asset_min_edge=0.0,
        asset_max_price=0.95,
        p_flip=dec2_res.p_flip,
        model_ver=dec2_res.model_ver,
    )
    await execute_and_record(
        db_session=db_session,
        market=mkt2,
        decision_obj=dec2_res.decision_obj,
        validation=val2,
        asset_mode="FAVORITE",
        active_features="ct_features",
        p_flip=dec2_res.p_flip,
        model_ver=dec2_res.model_ver,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        existing_skipped=None,
        start_time=dec_at,
        decision_at=dec_at,
    )
    await db_session.commit()

    # Verify request is in state READY
    req2_db = (await db_session.execute(select(ExecutionRequest).where(ExecutionRequest.market_id == mkt2.market_id))).scalar_one()
    assert req2_db.state == "READY"

    # Fresh worker session finds it via claim_one and executes
    import polyflip.execution.worker as worker_module
    bind = getattr(db_session, "bind", None) or (db_session.get_bind() if hasattr(db_session, "get_bind") else None)
    session_factory = async_sessionmaker(bind, expire_on_commit=False)
    orig_worker_session = worker_module.async_session
    worker_module.async_session = session_factory
    try:
        gateway2 = FakeExecutionGateway(profile="LIVE_PARITY", quote_provider=retry_quote_prov, fee_rate=Decimal("0.002"))
        await process_ready_requests(gateway=gateway2, quote_provider=retry_quote_prov)
    finally:
        worker_module.async_session = orig_worker_session

    trade2_db = (await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == mkt2.market_id))).scalar_one()
    await db_session.refresh(req2_db)
    assert req2_db.state in {"FILLED", "PARTIALLY_FILLED_FINAL"}
    assert trade2_db.position_status == "OPEN"
    assert math.isclose(float(trade2_db.entry_filled_shares), 4.99002, abs_tol=1e-4)

    # Point 3: Crash after fill before trade accounting update
    # Simulate: ExecutionFill written to DB, request FILLED, but worker crashed before rebuild_trade_accounting
    mkt3 = _make_live_market(db_session, market_id="mkt_crash_3", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()
    await _seed_market_snapshots(db_session, mkt3.market_id, dec_at, [0.20, 0.22, 0.20, 0.22, 0.20, 0.22, 0.20, 0.22])

    dec3_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=CrashClient(),
        market=mkt3,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at,
        time_left_sec=240.0,
        execution_mode="PAPER",
        decision_at=dec_at,
    )
    val3 = await validate_pre_trade(
        db_session=db_session,
        api_client=CrashClient(),
        market=mkt3,
        decision_obj=dec3_res.decision_obj,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        asset_mode="FAVORITE",
        asset_min_edge=0.0,
        asset_max_price=0.95,
        p_flip=dec3_res.p_flip,
        model_ver=dec3_res.model_ver,
    )
    await execute_and_record(
        db_session=db_session,
        market=mkt3,
        decision_obj=dec3_res.decision_obj,
        validation=val3,
        asset_mode="FAVORITE",
        active_features="ct_features",
        p_flip=dec3_res.p_flip,
        model_ver=dec3_res.model_ver,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        existing_skipped=None,
        start_time=dec_at,
        decision_at=dec_at,
    )
    await db_session.commit()

    req3 = await claim_one(db_session, "PAPER")
    assert req3 is not None
    trade3 = await db_session.get(TradeHistory, req3.trade_history_id)
    assert trade3 is not None

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
        gross_quote_usdc=Decimal("0.998004"),
        price=Decimal("0.20"),
        shares=Decimal("4.99002"),
        fee_usdc=Decimal("0.001996"),
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(fill3)
    req3.state = "FILLED"
    req3.filled_shares = Decimal("4.99002")
    req3.filled_cost_usdc = Decimal("0.998004")
    await db_session.commit()

    # Recovery: rebuild_trade_accounting restores position to OPEN
    await rebuild_trade_accounting(db_session, trade3.id)
    await db_session.commit()
    await db_session.refresh(trade3)

    assert trade3.position_status == "OPEN"
    assert math.isclose(float(trade3.entry_filled_shares), 4.99002, abs_tol=1e-5)
    assert math.isclose(float(trade3.remaining_shares), 4.99002, abs_tol=1e-5)
    assert math.isclose(float(trade3.entry_cost_usdc), 1.00000, abs_tol=1e-5)

    # Repeat call is an idempotent no-op!
    await rebuild_trade_accounting(db_session, trade3.id)
    await db_session.commit()
    await db_session.refresh(trade3)
    assert trade3.position_status == "OPEN"
    assert math.isclose(float(trade3.entry_filled_shares), 4.99002, abs_tol=1e-5)
    assert math.isclose(float(trade3.entry_cost_usdc), 1.00000, abs_tol=1e-5)


@pytest.mark.asyncio
async def test_13b_recovery_stuck_claimed_request(db_session, base_decision_time):
    """Requirement 19: Worker crash leaves request in CLAIMED with expired lease; fresh worker reclaims and completes it."""
    dec_at = base_decision_time
    now = datetime.now(timezone.utc)
    market = _make_live_market(db_session, market_id="mkt_stuck_claim", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()
    await _seed_market_snapshots(db_session, market.market_id, dec_at, [0.20, 0.22, 0.20, 0.22, 0.20, 0.22, 0.20, 0.22])

    class StuckClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    dec_res = await decide_ct_outsider_mode(
        db_session=db_session,
        api_client=StuckClient(),
        market=market,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        raw_settings={},
        models_cache=None,
        crypto_predictor=None,
        start_time=dec_at,
        time_left_sec=240.0,
        execution_mode="PAPER",
        decision_at=dec_at,
    )
    val = await validate_pre_trade(
        db_session=db_session,
        api_client=StuckClient(),
        market=market,
        decision_obj=dec_res.decision_obj,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        asset_mode="FAVORITE",
        asset_min_edge=0.0,
        asset_max_price=0.95,
        p_flip=dec_res.p_flip,
        model_ver=dec_res.model_ver,
    )
    await execute_and_record(
        db_session=db_session,
        market=market,
        decision_obj=dec_res.decision_obj,
        validation=val,
        asset_mode="FAVORITE",
        active_features="ct_features",
        p_flip=dec_res.p_flip,
        model_ver=dec_res.model_ver,
        cfg=parse_trading_settings({"TRADING_MODE": "ct_outsider"}),
        existing_skipped=None,
        start_time=dec_at,
        decision_at=dec_at,
    )
    await db_session.commit()

    req = await claim_one(db_session, "PAPER")
    assert req is not None
    trade = await db_session.get(TradeHistory, req.trade_history_id)
    assert trade is not None

    # Simulate crashed worker: lease expired 30 seconds ago
    req.state = "CLAIMED"
    req.claimed_by = "crashed-worker-pid-9999"
    req.claimed_at = now - timedelta(seconds=60)
    req.lease_expires_at = now - timedelta(seconds=30)
    await db_session.commit()

    # Fresh worker starts up and processes ready/expired requests
    import polyflip.execution.worker as worker_module
    bind = getattr(db_session, "bind", None) or (db_session.get_bind() if hasattr(db_session, "get_bind") else None)
    session_factory = async_sessionmaker(bind, expire_on_commit=False)
    orig_worker_session = worker_module.async_session
    worker_module.async_session = session_factory
    try:
        async def quote_prov(_tok: str):
            return {"asks": [{"price": 0.20, "size": 50.0}], "bids": [{"price": 0.19, "size": 50.0}], "best_ask": 0.20, "best_bid": 0.19}
        gateway = FakeExecutionGateway(profile="LIVE_PARITY", quote_provider=quote_prov, fee_rate=Decimal("0.002"))
        await process_ready_requests(gateway=gateway, quote_provider=quote_prov)
    finally:
        worker_module.async_session = orig_worker_session

    await db_session.refresh(req)
    await db_session.refresh(trade)
    assert req.state in {"FILLED", "PARTIALLY_FILLED_FINAL"}
    assert trade.position_status == "OPEN"
    assert math.isclose(float(trade.entry_filled_shares), 4.99002, abs_tol=1e-4)


@pytest.mark.asyncio
async def test_14_first_decision_immutability_on_quote_change(db_session, base_decision_time):
    """Requirement 20: First reserved decision is immutable; market quote changes on next poll do not override it."""
    dec_at = base_decision_time
    market = _make_live_market(db_session, market_id="mkt_immutable", end_time=dec_at + timedelta(seconds=240))
    await db_session.commit()
    await _seed_market_snapshots(db_session, market.market_id, dec_at, [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23])

    class UpClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    async def quote_prov(_tok: str):
        return {"asks": [{"price": 0.20, "size": 50.0}], "bids": [{"price": 0.19, "size": 50.0}], "best_ask": 0.20, "best_bid": 0.19}

    # First poll: UP is outsider (ask 0.20)
    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        UpClient(),
        dec_at,
        quote_provider=quote_prov,
        fee_rate=Decimal("0.002"),
    )
    assert trade is not None
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

    await _seed_market_snapshots(
        db_session,
        market.market_id,
        dec_at,
        [0.20, 0.22, 0.20, 0.22, 0.20, 0.22, 0.20, 0.22],
    )

    class TraceClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": dec_at - timedelta(seconds=1),
                "received_at": dec_at - timedelta(seconds=1),
            }

    async def trace_provider(_tok: str):
        return {
            "asks": [{"price": 0.20, "size": 100.0}],
            "bids": [{"price": 0.19, "size": 100.0}],
            "best_ask": 0.20,
            "best_bid": 0.19,
        }

    # 1. Full production paper cycle (dispatcher -> validator -> recorder -> worker)
    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        TraceClient(),
        dec_at,
        quote_provider=trace_provider,
        fee_rate=Decimal("0.002"),
    )
    assert trade is not None
    assert trade.position_status == "OPEN"

    # 2. Settlement: YES wins
    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="YES",
        payout_per_share=Decimal("1.0"),
    )
    await db_session.commit()
    await db_session.refresh(trade)

    # 3. Verify complete audit chain across real DB rows
    # A. Reservation row
    res = await get_ct_decision_reservation(db_session, f"CT:BTC_CT_T5_V1:{market.market_id}")
    assert res is not None
    assert res.trade_history_id == trade.id
    assert res.market_id == market.market_id
    res_dec_at = res.decision_at if res.decision_at.tzinfo is not None else res.decision_at.replace(tzinfo=timezone.utc)
    assert res_dec_at == dec_at

    # B. ExecutionRequest row
    req_stmt = select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade.id)
    req_db = (await db_session.execute(req_stmt)).scalar_one()
    assert req_db.state in {"FILLED", "PARTIALLY_FILLED_FINAL"}
    assert req_db.outcome_to_buy == "YES"
    assert math.isclose(float(req_db.filled_cost_usdc), 0.998004, abs_tol=1e-4)

    # C. ExecutionAttempt & ExecutionFill rows
    attempt_stmt = select(ExecutionAttempt).where(ExecutionAttempt.request_id == req_db.id)
    attempt_db = (await db_session.execute(attempt_stmt)).scalar_one()

    fill_stmt = select(ExecutionFill).where(ExecutionFill.attempt_id == attempt_db.id)
    fill_db = (await db_session.execute(fill_stmt)).scalar_one()
    assert fill_db.gateway == "FAKE"
    assert math.isclose(float(fill_db.price), 0.20, abs_tol=1e-5)
    assert math.isclose(float(fill_db.fee_usdc), 0.001996, abs_tol=1e-5)

    # D. TradeHistory row
    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    assert math.isclose(float(trade.entry_cost_usdc), 1.00000, abs_tol=1e-4)
    assert math.isclose(float(trade.realized_pnl_usdc), 3.99002, abs_tol=1e-4)

    # E. Full link verification
    assert res.key == f"CT:BTC_CT_T5_V1:{market.market_id}"
    assert req_db.trade_history_id == trade.id
    assert attempt_db.request_id == req_db.id
    assert fill_db.attempt_id == attempt_db.id
    assert trade.id == res.trade_history_id


def test_17_unassigned_parity_skips_do_not_pollute_down_side():
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
async def test_18_decision_runners_never_substitutes_ask_for_mid_when_bid_none(db_session):
    """Item 9 Self-check: Mid price is never substituted by ask when bid is missing."""
    from unittest.mock import AsyncMock
    from polyflip.trading.decision_runners import decide_ct_outsider_mode
    from polyflip.trading.trading_config import parse_trading_settings

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
        db_session=db_session,
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
async def test_19_market_guards_immutability_and_skip_reason_preservation():
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


@pytest.mark.asyncio
async def test_19b_market_guards_preserve_ct_skip_after_window():
    """A later heartbeat outside [T-300,T-210] must not overwrite CT's decision."""
    from unittest.mock import AsyncMock, MagicMock
    from polyflip.trading.market_guards import check_market_guards
    from polyflip.trading.trading_config import parse_trading_settings
    from polyflip.db.models import TradeHistory

    mock_db = AsyncMock()
    cfg = parse_trading_settings({"TRADING_MODE": "ct_outsider"})

    class DummyMarket:
        market_id = "mkt_immutable_outside"
        asset = "BTC"
        yes_token_id = "tok_yes"
        no_token_id = "tok_no"

    existing_skip = TradeHistory(
        market_id="mkt_immutable_outside",
        asset="BTC",
        status="SKIPPED",
        error_msg="REGIME_NOT_REVERSION: UNCERTAIN",
        strategy_name="BTC_CT_T5_V1",
        strategy_type="CT_OUTSIDER",
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_skip
    mock_db.execute = AsyncMock(return_value=mock_result)

    guard_res = await check_market_guards(
        db_session=mock_db,
        market=DummyMarket(),
        cfg=cfg,
        asset_mode="ct_outsider",
        time_left_sec=149.0,
        start_time=datetime.now(timezone.utc),
    )

    assert guard_res.passed is False
    assert guard_res.skip_reason == "guard: Decision already recorded in window (SKIP)"
    assert guard_res.existing_skipped is existing_skip
    assert existing_skip.error_msg == "REGIME_NOT_REVERSION: UNCERTAIN"


@pytest.mark.asyncio
async def test_20_decision_at_timing_fidelity(db_session, base_decision_time):
    """Requirement 4: Decision time is separate from cycle start, fixed after quotes arrive, and verified in DB and diagnostics."""
    cycle_start = base_decision_time
    # 300 ms simulated quote delay
    decision_at = cycle_start + timedelta(milliseconds=300)

    market = _make_live_market(db_session, market_id="mkt_timing_fid", end_time=cycle_start + timedelta(seconds=240))
    await db_session.commit()

    await _seed_market_snapshots(
        db_session,
        market.market_id,
        decision_at,
        [0.20, 0.22, 0.20, 0.22, 0.20, 0.22, 0.20, 0.22],
    )

    class TimingClient:
        async def get_market_prices(self, tok, **kwargs):
            return {
                "best_ask": 0.20,
                "best_bid": 0.19,
                "best_ask_no": 0.80,
                "best_bid_no": 0.79,
                "current_yes_price": 0.20,
                "current_no_price": 0.80,
                "event_at": decision_at - timedelta(seconds=1),
                "received_at": decision_at,
            }

    async def quote_prov(_tok: str):
        return {
            "asks": [{"price": 0.20, "size": 50.0}],
            "bids": [{"price": 0.19, "size": 50.0}],
            "best_ask": 0.20,
            "best_bid": 0.19,
        }

    trade, dec_res = await _run_production_paper_cycle(
        db_session,
        market,
        TimingClient(),
        cycle_start,
        quote_provider=quote_prov,
        decision_at=decision_at,
    )

    assert trade is not None
    assert dec_res is not None

    # Check reservation timestamp matches decision_at, NOT cycle_start!
    res = await get_ct_decision_reservation(db_session, f"CT:BTC_CT_T5_V1:{market.market_id}")
    assert res is not None
    res_dec_at = res.decision_at if res.decision_at.tzinfo is not None else res.decision_at.replace(tzinfo=timezone.utc)
    assert res_dec_at == decision_at
    assert res_dec_at != cycle_start
    assert (res_dec_at - cycle_start).total_seconds() == 0.3

    # Check timing diagnostics in decision details
    timing_diag = dec_res.decision_obj.decision_details.get("timing_diagnostics", {})
    assert timing_diag.get("cycle_started_at") == cycle_start.isoformat()
    assert timing_diag.get("decision_at") == decision_at.isoformat()
    # time_left_sec is computed relative to decision_at (240 - 0.3 = 239.7)
    expected_time_left = round((market.end_time_est - decision_at).total_seconds(), 3)
    assert math.isclose(float(timing_diag.get("time_left_sec")), expected_time_left, abs_tol=1e-2)
