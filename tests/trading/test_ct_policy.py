"""
tests/trading/test_ct_policy.py

Comprehensive test suite verifying Stages 1 and 2 of the CT Outsider Strategy:
- Spec BTC_CT_T5_V1 immutability, hashing, and parameter tamper detection.
- Unified CT calculation with causal bounds, NaN handling, and error containment.
- Explicit skips for INSUFFICIENT_HISTORY (<3), INVALID_SERIES, CALCULATION_ERROR.
- Symmetric outsider selection (UP vs DOWN) by mid price and PARITY handling.
- Distinct token histories (no 1 - YES reconstruction for DOWN).
- Policy symmetry tests (UP <-> DOWN mirroring).
- 3 diagnostic controls per market (YES-only, symmetric price, symmetric CT).
"""
from __future__ import annotations

import copy
import dataclasses
from datetime import datetime, timezone, timedelta
import math
import numpy as np
import pytest

from polyflip.trading.ct_policy import (
    CTSpecification,
    BTC_CT_T5_V1_SPEC,
    get_btc_ct_t5_v1_spec,
    compute_token_ct_regime,
    MarketTokenMapping,
    SideQuote,
    CTDecision,
    evaluate_ct_policy,
    evaluate_all_ct_diagnostics,
    calculate_scenario_economics,
)


# ==============================================================================
# 1. Specification & Hash Tests (Stage 1, Item 2)
# ==============================================================================

def test_spec_parameters_locked():
    """Verify BTC_CT_T5_V1 exact specification parameters."""
    spec = get_btc_ct_t5_v1_spec()
    assert spec.spec_id == "BTC_CT_T5_V1"
    assert spec.version == 1
    assert spec.execution_contour == "PAPER"
    assert spec.asset == "BTC"
    assert spec.contract_horizon_min == 15
    # Critical self-check: window is 210 to 300 seconds (not 270 to 300)
    assert spec.decision_window_min_sec == 210.0
    assert spec.decision_window_max_sec == 300.0
    assert spec.price_min == 0.01
    assert spec.price_max == 0.40
    assert spec.ct_history_window_sec == 900.0
    assert spec.min_observations == 3
    assert spec.required_signal == "REVERSION"
    assert spec.max_purchase_cost_usdc == 1.00
    assert spec.taker_fee_rate == 0.002
    assert spec.exit_policy == "HOLD_TO_SETTLEMENT"
    assert spec.max_decisions_per_market == 1


def test_spec_hash_changes_on_any_parameter_mutation():
    """Self-check Item 2: changing any parameter changes spec_hash."""
    base_spec = get_btc_ct_t5_v1_spec()
    base_hash = base_spec.spec_hash
    assert len(base_hash) == 64

    # Test mutating every critical field
    mutations = [
        {"decision_window_min_sec": 270.0},  # Tamper attempt: 270 instead of 210
        {"decision_window_max_sec": 305.0},
        {"price_max": 0.45},
        {"price_min": 0.02},
        {"min_observations": 4},
        {"max_purchase_cost_usdc": 2.0},
        {"taker_fee_rate": 0.001},
        {"required_signal": "TREND"},
        {"asset": "ETH"},
        {"execution_contour": "LIVE"},
        {"version": 2},
    ]

    for m in mutations:
        mutated_spec = dataclasses.replace(base_spec, **m)
        assert mutated_spec.spec_hash != base_hash, f"Hash did not change for mutation: {m}"


# ==============================================================================
# 2. Unified CT Calculation Function (Stage 1, Items 3 & 5)
# ==============================================================================

def test_compute_token_ct_regime_reversion_signal():
    """Generates a known saw/reversion sequence and verifies classification."""
    decision_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Sawtooth pattern (high sign flips, negative autocorrelation, low ER)
    prices = [0.20, 0.26, 0.19, 0.25, 0.20, 0.26, 0.19, 0.25]
    obs = [
        {"recorded_at": decision_at - timedelta(minutes=14 - i), "mid_price": p}
        for i, p in enumerate(prices)
    ]
    res = compute_token_ct_regime(obs, decision_at=decision_at)
    assert res.valid is True
    assert res.status == "VALID"
    assert res.state == "REVERSION"
    assert res.observations_count == 8


def test_compute_token_ct_regime_trend_signal():
    """Monotonic trend must classify as TREND, not REVERSION."""
    decision_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    prices = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]
    obs = [
        {"recorded_at": decision_at - timedelta(minutes=10 - i), "mid_price": p}
        for i, p in enumerate(prices)
    ]
    res = compute_token_ct_regime(obs, decision_at=decision_at)
    assert res.valid is True
    assert res.state == "TREND"


def test_compute_token_ct_regime_quiet_signal():
    """Flat prices must classify as QUIET, never REVERSION."""
    decision_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    prices = [0.25, 0.25, 0.25, 0.250001, 0.25]
    obs = [
        {"recorded_at": decision_at - timedelta(minutes=5 - i), "mid_price": p}
        for i, p in enumerate(prices)
    ]
    res = compute_token_ct_regime(obs, decision_at=decision_at)
    assert res.valid is True
    assert res.state == "QUIET"


def test_insufficient_history_explicit_skip():
    """Self-check Item 5: exactly 2 observations returns INSUFFICIENT_HISTORY."""
    decision_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = [
        {"recorded_at": decision_at - timedelta(minutes=5), "mid_price": 0.20},
        {"recorded_at": decision_at - timedelta(minutes=1), "mid_price": 0.22},
    ]
    res = compute_token_ct_regime(obs, decision_at=decision_at, min_observations=3)
    assert res.valid is False
    assert res.status == "INSUFFICIENT_HISTORY"
    assert res.state == "UNCERTAIN"


def test_nan_series_returns_invalid_series():
    """Self-check Item 5: NaN-only series returns INVALID_SERIES."""
    decision_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = [
        {"recorded_at": decision_at - timedelta(minutes=5), "mid_price": float("nan")},
        {"recorded_at": decision_at - timedelta(minutes=3), "mid_price": None},
        {"recorded_at": decision_at - timedelta(minutes=1), "mid_price": -0.1},
    ]
    res = compute_token_ct_regime(obs, decision_at=decision_at)
    assert res.valid is False
    assert res.status == "INVALID_SERIES"
    assert res.state == "UNCERTAIN"


def test_causality_boundary_filtering():
    """Self-check Item 11: future observations (> decision_at) are strictly excluded."""
    decision_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = [
        {"recorded_at": decision_at - timedelta(minutes=10), "mid_price": 0.20},
        {"recorded_at": decision_at - timedelta(minutes=5), "mid_price": 0.25},
        {"recorded_at": decision_at - timedelta(minutes=1), "mid_price": 0.20},
        # Future points that should be ignored
        {"recorded_at": decision_at + timedelta(seconds=1), "mid_price": 0.99},
        {"recorded_at": decision_at + timedelta(minutes=5), "mid_price": 0.99},
    ]
    res = compute_token_ct_regime(obs, decision_at=decision_at)
    assert res.observations_count == 3
    assert res.raw_prices == (0.20, 0.25, 0.20)
    assert 0.99 not in res.raw_prices


# ==============================================================================
# 3. Pure Policy Decision Tests (Stage 1 Item 4, Stage 2 Items 8-12)
# ==============================================================================

@pytest.fixture
def valid_market_setup():
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping(
        market_id="mkt_btc_123",
        asset="BTC",
        expiration=dec_at + timedelta(seconds=240),  # 240s left (in [210, 300])
        up_token_id="token_up_111",
        down_token_id="token_down_222",
    )
    # UP is outsider: mid 0.20 vs DOWN mid 0.80. Ask = 0.21 in [0.01, 0.40]
    up_quote = SideQuote(
        side="UP",
        token_id="token_up_111",
        best_bid=0.19,
        best_ask=0.21,
        mid_price=0.20,
        event_at=dec_at - timedelta(seconds=2),
    )
    down_quote = SideQuote(
        side="DOWN",
        token_id="token_down_222",
        best_bid=0.79,
        best_ask=0.81,
        mid_price=0.80,
        event_at=dec_at - timedelta(seconds=2),
    )
    # Reversion history for UP token
    reversion_prices = [0.18, 0.24, 0.17, 0.23, 0.18, 0.24, 0.19, 0.23]
    up_history = [
        {"recorded_at": dec_at - timedelta(minutes=14 - i), "mid_price": p}
        for i, p in enumerate(reversion_prices)
    ]
    return spec, dec_at, mapping, up_quote, down_quote, up_history


def test_evaluate_ct_policy_buy_up_outsider(valid_market_setup):
    """Happy path: UP outsider with valid ask and REVERSION history triggers BUY."""
    spec, dec_at, mapping, up_quote, down_quote, up_history = valid_market_setup
    decision = evaluate_ct_policy(spec, dec_at, mapping, up_quote, down_quote, up_history)

    assert decision.action == "BUY"
    assert decision.side == "UP"
    assert decision.token_id == "token_up_111"
    assert decision.reason == "CT_SIGNAL_REVERSION"
    assert decision.limit_price == 0.21
    assert decision.budget_usdc == 1.00
    assert decision.is_executable is True
    assert decision.ct_regime == "REVERSION"


def test_evaluate_ct_policy_buy_down_outsider():
    """Symmetric path: DOWN outsider with valid ask and REVERSION history triggers BUY."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping(
        market_id="mkt_btc_down_test",
        asset="BTC",
        expiration=dec_at + timedelta(seconds=250),
        up_token_id="token_up_111",
        down_token_id="token_down_222",
    )
    # DOWN is outsider: DOWN mid 0.25 vs UP mid 0.75. Down ask = 0.26
    up_quote = SideQuote(
        side="UP",
        token_id="token_up_111",
        best_bid=0.74,
        best_ask=0.76,
        mid_price=0.75,
        event_at=dec_at - timedelta(seconds=1),
    )
    down_quote = SideQuote(
        side="DOWN",
        token_id="token_down_222",
        best_bid=0.24,
        best_ask=0.26,
        mid_price=0.25,
        event_at=dec_at - timedelta(seconds=1),
    )
    down_history = [
        {"recorded_at": dec_at - timedelta(minutes=14 - i), "mid_price": p}
        for i, p in enumerate([0.22, 0.28, 0.21, 0.27, 0.23, 0.29, 0.22, 0.28])
    ]

    decision = evaluate_ct_policy(spec, dec_at, mapping, up_quote, down_quote, down_history)
    assert decision.action == "BUY"
    assert decision.side == "DOWN"
    assert decision.token_id == "token_down_222"
    assert decision.limit_price == 0.26
    assert decision.is_executable is True


def test_symmetry_policy_mirroring():
    """Self-check Item 12: Swapping UP and DOWN symmetrically mirrors the decision side."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping(
        market_id="mkt_symm",
        asset="BTC",
        expiration=dec_at + timedelta(seconds=270),
        up_token_id="token_up",
        down_token_id="token_down",
    )
    prices = [0.20, 0.26, 0.19, 0.25, 0.20, 0.26, 0.19, 0.25]
    history = [{"recorded_at": dec_at - timedelta(minutes=10 - i), "mid_price": p} for i, p in enumerate(prices)]

    # Case A: UP is outsider
    quote_a_up = SideQuote(side="UP", token_id="token_up", best_bid=0.19, best_ask=0.22, mid_price=0.20, event_at=dec_at - timedelta(seconds=1))
    quote_a_down = SideQuote(side="DOWN", token_id="token_down", best_bid=0.78, best_ask=0.81, mid_price=0.80, event_at=dec_at - timedelta(seconds=1))
    dec_a = evaluate_ct_policy(spec, dec_at, mapping, quote_a_up, quote_a_down, history)

    # Case B: Mirror quotes and pass history for DOWN
    quote_b_up = SideQuote(side="UP", token_id="token_up", best_bid=0.78, best_ask=0.81, mid_price=0.80, event_at=dec_at - timedelta(seconds=1))
    quote_b_down = SideQuote(side="DOWN", token_id="token_down", best_bid=0.19, best_ask=0.22, mid_price=0.20, event_at=dec_at - timedelta(seconds=1))
    dec_b = evaluate_ct_policy(spec, dec_at, mapping, quote_b_up, quote_b_down, history)

    assert dec_a.action == "BUY" and dec_b.action == "BUY"
    assert dec_a.side == "UP" and dec_b.side == "DOWN"
    assert dec_a.limit_price == dec_b.limit_price == 0.22
    assert dec_a.reason == dec_b.reason == "CT_SIGNAL_REVERSION"


def test_parity_skip():
    """Self-check Item 9: Identical mid prices return PARITY skip."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=250), "u", "d")
    q_up = SideQuote("UP", "u", 0.49, 0.51, 0.50, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.49, 0.51, 0.50, event_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert dec.reason == "PARITY"


def test_outside_decision_window_skip():
    """Window outside 210-300 returns OUTSIDE_WINDOW."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    # 350 seconds left (> 300)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=350), "u", "d")
    q_up = SideQuote("UP", "u", 0.19, 0.21, 0.20, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert dec.reason == "OUTSIDE_WINDOW"


def test_price_filter_skip():
    """Ask > 0.40 returns PRICE_FILTER."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    # Outsider ask is 0.45 (> 0.40)
    q_up = SideQuote("UP", "u", 0.39, 0.45, 0.42, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.55, 0.61, 0.58, event_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert dec.reason == "PRICE_FILTER"


def test_missing_token_history_for_down_outsider():
    """Self-check Item 10: Missing NO history for DOWN returns MISSING_TOKEN_HISTORY."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    # DOWN is outsider
    q_up = SideQuote("UP", "u", 0.75, 0.80, 0.78, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.20, 0.25, 0.22, event_at=dec_at - timedelta(seconds=1))

    # Empty history passed
    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, chosen_token_history=None)
    assert dec.action == "SKIP"
    assert dec.reason == "MISSING_TOKEN_HISTORY"
    assert dec.side == "DOWN"


def test_ambiguous_token_mapping_rejected():
    """Self-check Item 8: Duplicate or missing token IDs block decision."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Ambiguous: up_token_id == down_token_id
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "token_x", "token_x")
    q_up = SideQuote("UP", "token_x", 0.19, 0.21, 0.20, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "token_x", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert "MAPPING_ERROR" in dec.reason


def test_three_diagnostic_results_saved(valid_market_setup):
    """Self-check Item 13: 3 diagnostics recorded, only symmetric_ct_policy is executable."""
    spec, dec_at, mapping, up_quote, down_quote, up_history = valid_market_setup
    diag = evaluate_all_ct_diagnostics(
        spec=spec,
        decision_at=dec_at,
        market_mapping=mapping,
        up_quote=up_quote,
        down_quote=down_quote,
        up_history=up_history,
        down_history=[],
    )
    assert diag.market_id == mapping.market_id
    assert diag.historical_yes_only.is_executable is False
    assert diag.symmetric_price_control.is_executable is False
    assert diag.symmetric_ct_policy.is_executable is True
    assert diag.symmetric_ct_policy.action == "BUY"


# ==============================================================================
# 4. Economic Scenario Accounting Tests (Stage 1 Item 7)
# ==============================================================================

def test_calculate_scenario_economics_win():
    """1 USDC budget with ask 0.04 and 0.002 fee gives +23.998 USDC net PnL."""
    econ = calculate_scenario_economics(ask=0.04, outcome="WIN", budget_usdc=1.0, taker_fee_rate=0.002)
    assert math.isclose(econ["shares"], 25.0, abs_tol=1e-6)
    assert math.isclose(econ["gross_pnl"], 25.0 * (1.0 - 0.04), abs_tol=1e-6)  # 24.0
    assert math.isclose(econ["fee"], 0.002, abs_tol=1e-6)
    assert math.isclose(econ["net_pnl"], 23.998, abs_tol=1e-6)


def test_calculate_scenario_economics_loss():
    """1 USDC budget with ask 0.20 and 0.002 fee gives -1.002 USDC net PnL."""
    econ = calculate_scenario_economics(ask=0.20, outcome="LOSS", budget_usdc=1.0, taker_fee_rate=0.002)
    assert math.isclose(econ["shares"], 5.0, abs_tol=1e-6)
    assert math.isclose(econ["gross_pnl"], -1.0, abs_tol=1e-6)
    assert math.isclose(econ["fee"], 0.002, abs_tol=1e-6)
    assert math.isclose(econ["net_pnl"], -1.002, abs_tol=1e-6)


# ==============================================================================
# 5. Quote Timing and Causality Tests (Stage 3, Requirements 5-9)
# ==============================================================================

def test_quote_timing_fresh():
    """Requirement 9: Fresh quote within 15s is valid."""
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    q = SideQuote("UP", "u", 0.20, 0.22, 0.21, event_at=dec_at - timedelta(seconds=5.0))
    valid, err = q.is_valid_for_decision(dec_at, max_staleness_sec=15.0)
    assert valid is True
    assert err is None
    assert math.isclose(q.compute_age_sec(dec_at), 5.0, abs_tol=1e-6)


def test_quote_timing_stale_rejection():
    """Requirement 6, 9: Quote older than 15s is rejected with STALE_QUOTE."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    q_stale = SideQuote("UP", "u", 0.20, 0.22, 0.21, event_at=dec_at - timedelta(seconds=16.0))
    q_fresh = SideQuote("DOWN", "d", 0.78, 0.80, 0.79, event_at=dec_at - timedelta(seconds=2.0))

    valid, err = q_stale.is_valid_for_decision(dec_at, max_staleness_sec=15.0)
    assert valid is False
    assert err == "STALE_QUOTE"

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_stale, q_fresh, [])
    assert dec.action == "SKIP"
    assert "STALE_QUOTE" in dec.reason


def test_quote_timing_exact_boundary():
    """Requirement 9: Exact boundary: 15.0s is valid, 15.001s is STALE_QUOTE."""
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    q_15_0 = SideQuote("UP", "u", 0.20, 0.22, 0.21, event_at=dec_at - timedelta(seconds=15.0))
    valid_boundary, err_boundary = q_15_0.is_valid_for_decision(dec_at, max_staleness_sec=15.0)
    assert valid_boundary is True
    assert err_boundary is None

    q_15_001 = SideQuote("UP", "u", 0.20, 0.22, 0.21, event_at=dec_at - timedelta(seconds=15.001))
    valid_stale, err_stale = q_15_001.is_valid_for_decision(dec_at, max_staleness_sec=15.0)
    assert valid_stale is False
    assert err_stale == "STALE_QUOTE"


def test_quote_timing_missing_timestamp():
    """Requirement 5, 9: Quote without timing marks is rejected with MISSING_QUOTE_TIMESTAMP."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    q_no_ts = SideQuote("UP", "u", 0.20, 0.22, 0.21, event_at=None, received_at=None)
    q_valid = SideQuote("DOWN", "d", 0.78, 0.80, 0.79, event_at=dec_at - timedelta(seconds=2.0))

    valid, err = q_no_ts.is_valid_for_decision(dec_at)
    assert valid is False
    assert err == "MISSING_QUOTE_TIMESTAMP"
    assert q_no_ts.compute_age_sec(dec_at) is None

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_no_ts, q_valid, [])
    assert dec.action == "SKIP"
    assert "MISSING_QUOTE_TIMESTAMP" in dec.reason


def test_quote_timing_negative_age_future():
    """Requirement 7, 9: Negative age (future timestamp) triggers FUTURE_QUOTE_DETECTED."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")

    # Future event_at
    q_future_evt = SideQuote("UP", "u", 0.20, 0.22, 0.21, event_at=dec_at + timedelta(seconds=2.0))
    valid_evt, err_evt = q_future_evt.is_valid_for_decision(dec_at)
    assert valid_evt is False
    assert "FUTURE_QUOTE_DETECTED" in err_evt
    assert "EVENT_IN_FUTURE" in err_evt

    # Future received_at
    q_future_rec = SideQuote("UP", "u", 0.20, 0.22, 0.21, received_at=dec_at + timedelta(seconds=3.0))
    valid_rec, err_rec = q_future_rec.is_valid_for_decision(dec_at)
    assert valid_rec is False
    assert "FUTURE_QUOTE_DETECTED" in err_rec
    assert "RECEIVED_IN_FUTURE" in err_rec

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_future_evt, q_future_rec, [])
    assert dec.action == "SKIP"
    assert "FUTURE_QUOTE_DETECTED" in dec.reason


def test_quote_timing_received_vs_event_precedence():
    """Requirement 5, 9: event_at is preferred over received_at for age computation."""
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Both timestamps present: event_at (4s old) vs received_at (2s old)
    q_both = SideQuote(
        "UP", "u", 0.20, 0.22, 0.21,
        event_at=dec_at - timedelta(seconds=4.0),
        received_at=dec_at - timedelta(seconds=2.0),
    )
    # compute_age_sec must use event_at (4.0s)
    assert math.isclose(q_both.compute_age_sec(dec_at), 4.0, abs_tol=1e-6)

    # Only received_at present: fallback to received_at (2.0s)
    q_rec_only = SideQuote(
        "UP", "u", 0.20, 0.22, 0.21,
        event_at=None,
        received_at=dec_at - timedelta(seconds=2.0),
    )
    assert math.isclose(q_rec_only.compute_age_sec(dec_at), 2.0, abs_tol=1e-6)


def test_quote_with_future_received_at_rejected():
    """Item 11: Quote with received_at > decision_at must be rejected as FUTURE_QUOTE_RECEIVED."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    q_up = SideQuote("UP", "u", 0.19, 0.21, 0.20, event_at=dec_at - timedelta(seconds=1), received_at=dec_at + timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=1), received_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert "FUTURE_QUOTE_RECEIVED" in dec.reason


def test_quote_with_crossed_book_rejected():
    """Item 9: Quote with best_bid > best_ask must be rejected as CROSSED_BOOK_QUOTE."""
    spec = get_btc_ct_t5_v1_spec()
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "BTC", dec_at + timedelta(seconds=240), "u", "d")
    # Crossed book: bid 0.25 > ask 0.20
    q_up = SideQuote("UP", "u", 0.25, 0.20, 0.22, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert "CROSSED_BOOK_QUOTE" in dec.reason


def test_asset_mismatch_rejected():
    """Item 8: Asset mismatch against specification must return ASSET_MISMATCH skip."""
    spec = get_btc_ct_t5_v1_spec()  # asset is BTC
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mapping = MarketTokenMapping("mkt", "ETH", dec_at + timedelta(seconds=240), "u", "d")
    q_up = SideQuote("UP", "u", 0.19, 0.21, 0.20, event_at=dec_at - timedelta(seconds=1))
    q_down = SideQuote("DOWN", "d", 0.79, 0.81, 0.80, event_at=dec_at - timedelta(seconds=1))

    dec = evaluate_ct_policy(spec, dec_at, mapping, q_up, q_down, [])
    assert dec.action == "SKIP"
    assert "ASSET_MISMATCH" in dec.reason


def test_causality_and_audit_metadata_in_data_ids(valid_market_setup):
    """Item 11 & 25: Decision must contain opportunity_id, decision_id, and quote age/receipt timestamps."""
    spec, dec_at, mapping, up_quote, down_quote, up_history = valid_market_setup
    decision = evaluate_ct_policy(spec, dec_at, mapping, up_quote, down_quote, up_history)

    assert decision.action == "BUY"
    d = decision.data_ids
    assert d["opportunity_id"] == f"{mapping.market_id}_{dec_at.isoformat()}"
    assert d["decision_id"] == f"CT:{spec.spec_id}:{mapping.market_id}"
    assert "up_quote_age_sec" in d
    assert "down_quote_age_sec" in d
    assert d["up_quote_age_sec"] is not None and d["up_quote_age_sec"] >= 0.0


def test_compute_token_ct_regime_matches_classify_local_regime_directly():
    """Item 3 Self-check: Direct classify_local_regime and compute_token_ct_regime yield identical states."""
    from polyflip.research.regime_features import classify_local_regime
    dec_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Sawtooth reversion
    rev_prices = [0.20, 0.26, 0.19, 0.25, 0.20, 0.26, 0.19, 0.25]
    obs_rev = [{"recorded_at": dec_at - timedelta(minutes=10 - i), "mid_price": p} for i, p in enumerate(rev_prices)]
    res_rev = compute_token_ct_regime(obs_rev, decision_at=dec_at, min_observations=3)
    clf_rev = classify_local_regime(rev_prices, min_observations=3)
    assert res_rev.state == clf_rev["state"] == "REVERSION"

    # 2. Monotonic trend
    trend_prices = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]
    obs_tr = [{"recorded_at": dec_at - timedelta(minutes=10 - i), "mid_price": p} for i, p in enumerate(trend_prices)]
    res_tr = compute_token_ct_regime(obs_tr, decision_at=dec_at, min_observations=3)
    clf_tr = classify_local_regime(trend_prices, min_observations=3)
    assert res_tr.state == clf_tr["state"] == "TREND"

    # 3. Flat quiet
    quiet_prices = [0.25, 0.25, 0.25, 0.25, 0.25]
    obs_q = [{"recorded_at": dec_at - timedelta(minutes=10 - i), "mid_price": p} for i, p in enumerate(quiet_prices)]
    res_q = compute_token_ct_regime(obs_q, decision_at=dec_at, min_observations=3)
    clf_q = classify_local_regime(quiet_prices, min_observations=3)
    assert res_q.state == clf_q["state"] == "QUIET"
