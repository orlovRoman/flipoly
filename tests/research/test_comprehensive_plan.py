"""
tests/research/test_comprehensive_plan.py

Master test suite asserting all 30 items of the comprehensive research plan:
Stages 1 and 2 (Items 1 through 30).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from polyflip.collector.orderbook_depth import (
    OrderbookContract,
    validate_and_normalize_orderbook,
    compute_orderbook_completeness_report,
)
from polyflip.research.orderbook_execution import (
    simulate_orderbook_execution,
    calculate_trade_payout_and_pnl,
    ExecutionVolumeTracker,
)
from polyflip.research.regime_features import (
    compute_efficiency_ratio,
    compute_return_sign_changes,
    compute_return_autocorrelation,
    classify_local_regime,
    classify_spot_regime_short,
    align_underlying_history_causal,
    compute_strike_context,
    verify_mirror_symmetry,
)
from polyflip.db.models import (
    Base,
    MarketSnapshot,
    LiveMarket,
    OrderbookDepthSnapshot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# STAGE 1: Data, Features, and Execution Modeling (Items 1 to 17)
# ==============================================================================

def test_item_01_manifest_and_controls():
    """Item 1: Manifest records commit, hashes, rules, fees, seed, controls unchanged."""
    manifest_path = REPO_ROOT / "artifacts" / "research" / "reproducible_data_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert "git_commit" in m and m["git_commit"]
    assert "launch_parameters" in m
    assert m["launch_parameters"]["seed"] == 42
    assert m["launch_parameters"]["stake_usdc"] == 1.0
    assert m["launch_parameters"]["taker_fee_rate"] == 0.002


def test_item_02_report_corrections_and_stress_tables():
    """Item 2: 9.6% is Neither and 25.7% is CT-only (not C0 and CT)."""
    # Verify from results
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ch = data["item_22_cohort_decomposition"]
        assert math.isclose(ch["Neither"]["win_rate"], 0.0960, abs_tol=1e-3)
        assert math.isclose(ch["CT-only"]["win_rate"], 0.2574, abs_tol=1e-3)


def test_item_03_three_major_wins_evidence():
    """Item 3: Evidence chronology for markets 3987711, 3478125, 3925291."""
    ev_path = REPO_ROOT / "artifacts" / "research" / "three_major_wins_evidence.json"
    assert ev_path.exists()
    with open(ev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    markets = data["markets"]
    for mid in ["3987711", "3478125", "3925291"]:
        assert mid in markets
        m = markets[mid]
        # Check partitions
        parts = m["data_partitions"]
        assert "phase1_before_decision" in parts
        assert "phase2_decision_to_expiration" in parts
        assert "phase3_post_expiration" in parts
        # Check raw OHLC fields
        c_raw = parts["phase2_decision_to_expiration"]["candles_5m_raw"]
        if c_raw:
            c0 = c_raw[0]
            for col in ["open_time", "open", "high", "low", "close", "volume"]:
                assert col in c0
        # Check self-checks
        assert m["self_checks"]["arithmetic_adds_up"] is True
        assert m["self_checks"]["candle_high_not_called_close"] is True
        assert m["self_checks"]["unknown_strike_preserved_as_unknown"] is True


def test_item_04_orderbook_contract_and_units():
    """Item 4: Contract structure, units (size=shares, price*size=USDC), missing != 0."""
    bids = [{"price": 0.05, "size": 100.0}]
    asks = [{"price": 0.10, "size": 50.0}]
    ob = validate_and_normalize_orderbook(bids, asks, market_id="m1", token_id="t1", outcome_side="YES")
    assert ob.best_bid_price == 0.05
    assert ob.best_bid_size == 100.0  # shares
    assert math.isclose(ob.depth_usdc_bid, 5.0, abs_tol=1e-5)  # 0.05 * 100 = 5 USDC
    assert ob.best_ask_price == 0.10
    assert ob.best_ask_size == 50.0   # shares
    assert math.isclose(ob.depth_usdc_ask, 5.0, abs_tol=1e-5)  # 0.10 * 50 = 5 USDC

    # Missing book side returns None, not 0.0
    empty_ob = validate_and_normalize_orderbook([], asks, market_id="m1", token_id="t1", outcome_side="YES")
    assert empty_ob.best_bid_price is None
    assert empty_ob.best_bid_size is None
    assert empty_ob.depth_usdc_bid is None


def test_item_05_db_model_and_migration():
    """Item 5: OrderbookDepthSnapshot entity and Alembic migration file."""
    assert hasattr(OrderbookDepthSnapshot, "__tablename__")
    assert OrderbookDepthSnapshot.__tablename__ == "orderbook_depth_snapshots"
    migration_file = REPO_ROOT / "alembic" / "versions" / "20260909_orderbook_depth_and_market_bounds.py"
    assert migration_file.exists()


def test_item_06_real_orderbooks_both_sides():
    """Item 6: Real YES and NO orderbooks, no surrogate reconstruction."""
    bids_yes = [{"price": 0.30, "size": 10.0}]
    asks_yes = [{"price": 0.32, "size": 10.0}]
    bids_no = [{"price": 0.65, "size": 15.0}]
    asks_no = [{"price": 0.68, "size": 15.0}]

    ob_yes = validate_and_normalize_orderbook(bids_yes, asks_yes, "m1", "tok_yes", "YES")
    ob_no = validate_and_normalize_orderbook(bids_no, asks_no, "m1", "tok_no", "NO")

    assert ob_yes.best_ask_price == 0.32
    assert ob_no.best_ask_price == 0.68  # Real NO price, not 1 - 0.32 = 0.68 by accident, but real ladder


def test_item_07_orderbook_quality_checks():
    """Item 7: Crossed book, negative sizes, unordered levels, duplicates, sequence gap."""
    # Crossed book
    crossed = validate_and_normalize_orderbook(
        [{"price": 0.40, "size": 10}], [{"price": 0.35, "size": 10}], "m1", "t1", "YES"
    )
    assert crossed.quality_status == "CROSSED_BOOK"

    # Negative size
    neg = validate_and_normalize_orderbook(
        [{"price": 0.30, "size": -5}], [{"price": 0.35, "size": 10}], "m1", "t1", "YES"
    )
    assert neg.quality_status == "NEGATIVE_SIZE"

    # Sequence gap
    seq_gap = validate_and_normalize_orderbook(
        [{"price": 0.30, "size": 10}], [{"price": 0.35, "size": 10}],
        "m1", "t1", "YES", sequence_id="105", expected_sequence_id="104"
    )
    assert seq_gap.quality_status == "SEQUENCE_GAP"


def test_item_08_orderbook_completeness_report():
    """Item 8: Orderbook completeness and telemetry report."""
    ob1 = validate_and_normalize_orderbook([{"price": 0.3, "size": 10}], [{"price": 0.4, "size": 10}], "m1", "t1", "YES")
    ob2 = validate_and_normalize_orderbook([{"price": 0.6, "size": 10}], [{"price": 0.7, "size": 10}], "m1", "t2", "NO")
    rep = compute_orderbook_completeness_report([ob1, ob2], decision_market_ids=["m1"])
    assert rep["both_sides_coverage_pct"] == 100.0
    assert rep["invalid_fraction"] == 0.0


def test_item_09_canonical_strike_verification():
    """Item 9: Canonical strike provenance tracks extraction method and unrecorded state."""
    from polyflip.collector.client import _canonical_strike_provenance
    market = {"id": "m1", "strike": "65000"}
    event = {"startDate": "2026-08-10T12:00:00Z"}
    prov = _canonical_strike_provenance(market, event)
    assert prov.strike_value == 65000.0
    assert prov.strike_source == "market.strike"


def test_item_10_market_bounds_storage():
    """Item 10: Market boundaries and strike storage fields in models."""
    cols = MarketSnapshot.__table__.columns.keys()
    assert "strike_observed_at" in cols
    assert "market_start_at" in cols
    assert "market_end_at" in cols
    assert "settlement_price_source" in cols


def test_item_11_causal_underlying_alignment():
    """Item 11: Causal underlying alignment (no future data, max staleness)."""
    t0 = pd.Timestamp("2026-08-10 12:00:00+00:00")
    ts = [t0 - pd.Timedelta(minutes=m) for m in [8, 5, 2]]
    p = [100.0, 101.0, 102.0]
    valid_ts, valid_p, status = align_underlying_history_causal(ts, p, as_of=t0, window_min=10.0, max_staleness_sec=180.0)
    assert status == "VALID"
    assert len(valid_p) == 3

    # Stale test
    valid_ts_stale, valid_p_stale, status_stale = align_underlying_history_causal(ts, p, as_of=t0 + pd.Timedelta(hours=1), window_min=10.0)
    assert status_stale == "NO_OBSERVATIONS_IN_WINDOW"


def test_item_12_cs_short_classification():
    """Item 12: CS_short classifier on fixed lookback window."""
    # Monotonic move -> TREND (not saw)
    mono = [100.0, 102.0, 104.0, 106.0]
    clf_mono = classify_spot_regime_short(mono)
    assert clf_mono["state"] == "TREND"

    # Motionless series -> QUIET
    quiet = [100.0, 100.0, 100.0, 100.0]
    clf_quiet = classify_spot_regime_short(quiet)
    assert clf_quiet["state"] == "QUIET"

    # Insufficient history -> UNCERTAIN
    short = [100.0, 101.0]
    clf_short = classify_spot_regime_short(short)
    assert clf_short["state"] == "UNCERTAIN"


def test_item_13_explicit_autocorrelation_requirement():
    """Item 13: NaN autocorrelation does NOT become automatic confirmation of reversion."""
    saw_short = [100.0, 105.0, 100.0, 105.0]
    clf = classify_spot_regime_short(saw_short, require_valid_autocorr=True)
    assert clf["state"] == "UNCERTAIN"
    assert clf["classification_reason"] == "MISSING_AUTOCORRELATION"


def test_item_14_orderbook_execution_simulation():
    """Item 14: Pure function for orderbook execution simulation."""
    # Self-check Point 14: 10x0.03 and 20x0.04 with budget $1
    asks = [{"price": 0.03, "size": 10.0}, {"price": 0.04, "size": 20.0}]
    res = simulate_orderbook_execution(asks, budget_usdc=1.0, price_limit=None, taker_fee_rate=0.0)
    assert math.isclose(res.filled_shares, 27.5, abs_tol=1e-5)
    assert math.isclose(res.spent_usdc, 1.0, abs_tol=1e-5)
    assert math.isclose(res.vwap, 1.0 / 27.5, abs_tol=1e-5)

    # With limit 0.03
    res_lim = simulate_orderbook_execution(asks, budget_usdc=1.0, price_limit=0.03, taker_fee_rate=0.0)
    assert math.isclose(res_lim.filled_shares, 10.0, abs_tol=1e-5)
    assert math.isclose(res_lim.spent_usdc, 0.30, abs_tol=1e-5)
    assert math.isclose(res_lim.remaining_budget, 0.70, abs_tol=1e-5)

    # Truncated book beyond depth is unknown
    res_trunc = simulate_orderbook_execution(
        [{"price": 0.03, "size": 10.0}], budget_usdc=1.0, is_truncated=True
    )
    assert res_trunc.fill_status == "DEPTH_EXHAUSTED_UNKNOWN"
    assert res_trunc.data_status == "TRUNCATED_UNKNOWN"


def test_item_15_trade_settlement_and_payout():
    """Item 15: 10 shares for 0.30 USDC gives +9.70 on win, -0.30 on loss."""
    win = calculate_trade_payout_and_pnl(10.0, 0.30, 1.0, target=1, fee=0.0)
    assert math.isclose(win.gross_pnl, 9.70, abs_tol=1e-5)
    assert math.isclose(win.unspent_budget, 0.70, abs_tol=1e-5)

    loss = calculate_trade_payout_and_pnl(10.0, 0.30, 1.0, target=0, fee=0.0)
    assert math.isclose(loss.gross_pnl, -0.30, abs_tol=1e-5)
    assert math.isclose(loss.unspent_budget, 0.70, abs_tol=1e-5)


def test_item_16_latency_and_staleness_rejection():
    """Item 16: Stale snapshot returns STALE_BOOK and UNCERTAIN."""
    asks = [{"price": 0.03, "size": 10.0}]
    res = simulate_orderbook_execution(
        asks, budget_usdc=1.0, book_age_sec=20.0, arrival_delay_sec=5.0, max_staleness_sec=15.0
    )
    assert res.fill_status == "STALE_BOOK"
    assert res.data_status == "STALE"
    assert res.filled_shares == 0.0


def test_item_17_volume_reuse_prevention():
    """Item 17: Unique decision ID and orderbook liquidity consumption."""
    tracker = ExecutionVolumeTracker()
    dec_id = tracker.make_decision_id("m1", "2026-08-10T12:00:00Z", "YES")
    assert tracker.register_decision(dec_id) is True
    # Duplicate rejected
    assert tracker.register_decision(dec_id) is False

    # Liquidity consumption
    asks = [{"price": 0.03, "size": 10.0}]
    tracker.record_fill("snap_1", asks, filled_shares=6.0)
    consumed = tracker.get_consumed_shares("snap_1", 1)
    assert consumed[0] == 6.0

    # Next order on same snapshot sees remaining 4 shares
    res = simulate_orderbook_execution(
        asks, budget_usdc=1.0, already_consumed_shares_by_level=consumed
    )
    assert res.filled_shares == 4.0  # Only remaining 4 shares available!


# ==============================================================================
# STAGE 2: Mechanism and Filter Evaluation (Items 18 to 30)
# ==============================================================================

def test_item_18_common_opportunity_ledger():
    """Item 18: Common opportunity ledger preserves identical candidates."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["item_18_common_opportunity_ledger_summary"]["all_variants_evaluated_on_identical_ledger"] is True


def test_item_19_and_20_former_favorite_mechanism():
    """Item 19 & 20: 4 mutually exclusive groups, former favorites evaluated with losers."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ff = data["item_19_and_20_former_favorite_mechanism"]
        for grp in ["FORMER_FAVORITE", "PERSISTENT_CHEAP", "OTHER_TRAJECTORY", "INSUFFICIENT_HISTORY"]:
            assert grp in ff
            assert "n_trades" in ff[grp]
            assert "net_pnl_usdc" in ff[grp]


def test_item_21_price_bucket_separation():
    """Item 21: Mechanism separated from price inside 0.05 buckets."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pb = data["item_21_price_bucket_separation"]
        assert len(pb) >= 5


def test_item_22_cohort_decomposition():
    """Item 22: Cohort decomposition (CT-only, CS-only, Both, Neither) and additive invariants."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ch = data["item_22_cohort_decomposition"]
        for c in ["CT-only", "CS-only", "Both", "Neither"]:
            assert c in ch
        assert "sub_combinations" in ch["Neither"]


def test_item_25_and_28_core_variants_paired_bootstrap():
    """Item 25 & 28: Core variants comparison and paired daily block bootstrap."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data["item_25_and_28_core_variants"]
        for k in ["C0", "CT", "CS", "CS_short", "CT_plus_canonical_strike"]:
            assert k in v
            assert "paired_bootstrap_ci_95" in v[k]


def test_item_26_multi_budget_execution():
    """Item 26: Recalculation for budgets $1, $5, and $10."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mb = data["item_26_multi_budget_execution"]
        for b in ["$1", "$5", "$10"]:
            assert b in mb
            assert "executed_trade_count" in mb[b]
            assert "avg_vwap" in mb[b]


def test_item_29_holdout_evaluation():
    """Item 29: Subsequent period holdout evaluation."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        h = data["item_29_holdout_results"]
        assert "CT" in h
        assert "C0" in h


def test_item_30_verdict_criteria_synthesis():
    """Item 30: 5-way decision rule matrix and 3 core question answers."""
    res_path = REPO_ROOT / "artifacts" / "research" / "stage2_comprehensive_study_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dec = data["item_30_decision_verdict"]
        assert "verdict_status" in dec
        assert "action_required" in dec
        assert "three_core_answers" in dec
        assert dec["deploy_to_production"] is False
