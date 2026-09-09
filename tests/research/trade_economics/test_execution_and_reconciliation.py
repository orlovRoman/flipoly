"""
Unit tests for execution analysis, causal matching, waterfall invariants, and breakeven math.
"""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from polyflip.research.trade_economics.commissions import calculate_commission_decimal, calculate_commission
from polyflip.research.trade_economics.pnl import calculate_fill_position, calculate_net_pnl
from polyflip.research.trade_economics.execution_analysis import (
    compute_diagnostic_spread,
    build_waterfall_decomposition,
    build_scenario_waterfall,
    build_matched_subset_waterfall
)
from polyflip.research.trade_economics.strategy_evaluation import compute_breakeven_thresholds, run_paired_daily_bootstrap
from polyflip.research.trade_economics.db_loader import match_opportunities_with_fills

def test_decimal_rounding_round_half_up():
    scheme = {
        "formula_id": "FIXED_PERCENTAGE",
        "parameters": {"rate": 0.001},
        "evidence_status": "DEMONSTRATION",
        "round_method": "ROUND_HALF_UP",
        "round_decimals": 4
    }
    # 0.60 * 1 * 0.001 = 0.0006
    d_fee = calculate_commission_decimal(scheme, "taker", "0.60", "1.0")
    assert d_fee == Decimal("0.0006")
    assert float(d_fee) == 0.0006

def test_diagnostic_spread_calculation():
    df = pd.DataFrame([{
        "best_bid": 0.40,
        "best_ask": 0.60,
        "executable_ask": 0.60,
        "shares": 10.0,
        "gross_pnl": 4.0,
        "net_pnl": 3.88
    }])
    res = compute_diagnostic_spread(df)
    # mid = 0.50, half_spread = 0.10, cost = 1.0
    assert abs(res["diagnostic_mid"].iloc[0] - 0.50) < 1e-6
    assert abs(res["diagnostic_half_spread"].iloc[0] - 0.10) < 1e-6
    assert abs(res["diagnostic_spread_cost"].iloc[0] - 1.0) < 1e-6
    # Original net_pnl is unmodified
    assert res["net_pnl"].iloc[0] == 3.88

def test_multi_fill_vwap_and_fee():
    fills = [
        {"shares": 10.0, "price": 0.40, "fee_usdc": 0.008},
        {"shares": 10.0, "price": 0.60, "fee_usdc": 0.012}
    ]
    pos = calculate_fill_position(fills)
    assert pos["filled_shares"] == 20.0
    assert pos["purchase_cash"] == 10.0
    assert pos["vwap"] == 0.50
    assert pos["total_fee_usdc"] == 0.02
    assert pos["fill_count"] == 2

def test_causal_matching_rules():
    opps = [
        {
            "opportunity_id": "opp_1",
            "market_id": "m1",
            "decision_at": "2026-08-01T12:00:00+00:00",
            "executable_ask": 0.50,
            "target": 1,
            "shares": 2.0,
            "variants": {"C0": True, "CT": True}
        }
    ]
    db_fills = pd.DataFrame([
        {
            "request_id": "req_wrong_dir",
            "market_id": "m1",
            "outcome_to_buy": "NO",
            "requested_mode": "LIVE",
            "request_created_at": "2026-08-01T12:00:10+00:00",
            "state": "FILLED",
            "fill_id": "f1",
            "fill_price": 0.50,
            "fill_shares": 2.0,
            "fill_fee": 0.0,
            "fill_timestamp": "2026-08-01T12:00:10+00:00",
            "gateway": "POLYMARKET"
        },
        {
            "request_id": "req_prior",
            "market_id": "m1",
            "outcome_to_buy": "YES",
            "requested_mode": "LIVE",
            "request_created_at": "2026-08-01T11:55:00+00:00",
            "state": "FILLED",
            "fill_id": "f2",
            "fill_price": 0.50,
            "fill_shares": 2.0,
            "fill_fee": 0.0,
            "fill_timestamp": "2026-08-01T11:55:00+00:00",
            "gateway": "POLYMARKET"
        }
    ])
    db_fills["request_created_at"] = pd.to_datetime(db_fills["request_created_at"], utc=True)
    db_fills["fill_timestamp"] = pd.to_datetime(db_fills["fill_timestamp"], utc=True)
    
    live_df, paper_df, unmatched_df, summary = match_opportunities_with_fills(opps, db_fills, causality_window_sec=120)
    assert len(live_df) == 0
    assert len(unmatched_df) == 1
    assert unmatched_df["reason"].iloc[0] == "TIMING_MISMATCH_PRIOR"
    assert summary["applicable_live_coverage_c0_ct"] == 0

def test_waterfall_invariants():
    df_trades = pd.DataFrame([
        {"shares": 100.0, "executable_ask": 0.50, "target": 1},
        {"shares": 100.0, "executable_ask": 0.50, "target": 0},
    ])
    wf = build_waterfall_decomposition("C0", df_trades, matched_fills_df=None, scenario_fee_rate=0.001)
    assert len(wf) == 5
    assert wf.loc[wf["step_number"] == 0, "cumulative_pnl_usdc"].iloc[0] == -0.20
    assert wf.loc[wf["step_number"] == 1, "cumulative_pnl_usdc"].iloc[0] == -0.10
    assert wf.loc[wf["step_number"] == 1, "incremental_change_usdc"].iloc[0] == 0.10

def test_breakeven_math():
    df = pd.DataFrame([
        {"shares": 10.0, "executable_ask": 0.50, "target": 1}, # cash=5, payout=10 -> gross=+5
        {"shares": 10.0, "executable_ask": 0.50, "target": 0}, # cash=5, payout=0  -> gross=-5
        {"shares": 10.0, "executable_ask": 0.50, "target": 1}, # cash=5, payout=10 -> gross=+5
    ])
    be = compute_breakeven_thresholds(df, baseline_fee_rate=0.002)
    assert be["n_trades"] == 3
    assert abs(be["breakeven_fee_rate"] - (5.0 / 15.0)) < 1e-4
    assert abs(be["breakeven_slippage_per_share_usdc"] - (4.97 / 30.0)) < 1e-4
    assert "breakeven_slippage_fixed_budget_usdc" in be
    assert "simple_average_entry_price" in be
    assert "weighted_average_entry_price" in be

def test_breakeven_fixed_budget_vs_linear():
    df = pd.DataFrame([
        {"shares": 2.0, "executable_ask": 0.50, "target": 1}, # budget=1.0, win
        {"shares": 5.0, "executable_ask": 0.20, "target": 0}, # budget=1.0, loss
        {"shares": 4.0, "executable_ask": 0.25, "target": 1}, # budget=1.0, win
    ])
    be = compute_breakeven_thresholds(df, baseline_fee_rate=0.002)
    s_fix = be["breakeven_slippage_fixed_budget_usdc"]
    # Check that fixed budget slippage sets PnL to ~0 under dynamic shares
    pnl_at_s = (1.0 / (0.50 + s_fix) - 1.002) + (0.0 - 1.002) + (1.0 / (0.25 + s_fix) - 1.002)
    assert abs(pnl_at_s) < 1e-4
    # Simple average ask: (0.50 + 0.20 + 0.25) / 3 = 0.3167
    assert abs(be["simple_average_entry_price"] - 0.3167) < 1e-3
    # Weighted average ask: 3.0 / 11.0 = 0.2727
    assert abs(be["weighted_average_entry_price"] - (3.0 / 11.0)) < 1e-4

def test_matched_subset_waterfall_invariants():
    matched_df = pd.DataFrame([
        {
            "is_ct": True,
            "decision_ask": 0.20,
            "hypothetical_shares": 5.0,
            "hypothetical_net_pnl": 3.998, # target=1: 5 - 1 - 0.002
            "filled_shares": 4.0,
            "vwap": 0.25,
            "slippage_cash": (0.25 - 0.20) * 4.0, # 0.20
            "fee_usdc": 0.0,
            "actual_net_pnl": 4.0 - 1.0 - 0.0 # 3.0
        }
    ])
    wf = build_matched_subset_waterfall("CT", matched_df)
    assert len(wf) == 5
    assert wf.loc[wf["step_number"] == 0, "cumulative_pnl_usdc"].iloc[0] == 4.00
    assert wf.loc[wf["step_number"] == 4, "cumulative_pnl_usdc"].iloc[0] == 3.00

def test_bootstrap_reproducibility():
    df_c0 = pd.DataFrame([
        {"calendar_date": "2026-08-01", "net_pnl": -1.0},
        {"calendar_date": "2026-08-02", "net_pnl": 2.0},
    ])
    df_ct = pd.DataFrame([
        {"calendar_date": "2026-08-01", "net_pnl": 0.5},
        {"calendar_date": "2026-08-02", "net_pnl": 3.0},
    ])
    b1 = run_paired_daily_bootstrap(df_c0, df_ct, n_bootstrap=100, seed=42)
    b2 = run_paired_daily_bootstrap(df_c0, df_ct, n_bootstrap=100, seed=42)
    assert b1["c0_net_pnl_ci95"] == b2["c0_net_pnl_ci95"]
    assert b1["delta_pnl_ci95"] == b2["delta_pnl_ci95"]
    assert b1["nonpositive_delta_count"] == b2["nonpositive_delta_count"]