"""
Unit tests for execution analysis, causal matching, waterfall invariants, and breakeven math.
"""
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from polyflip.research.trade_economics.commissions import calculate_commission_decimal, calculate_commission
from polyflip.research.trade_economics.pnl import calculate_fill_position, calculate_net_pnl
from polyflip.research.trade_economics.execution_analysis import compute_diagnostic_spread, build_waterfall_decomposition
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
    # DB fills:
    # 1. Wrong direction (NO)
    # 2. Prior timing (-300s)
    # 3. Valid YES (+10s)
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
    # Both DB fills are rejected: req_wrong_dir is NO, req_prior is 5 min prior
    assert len(live_df) == 0
    assert len(unmatched_df) == 1
    assert unmatched_df["reason"].iloc[0] == "TIMING_MISMATCH_PRIOR"

def test_waterfall_invariants():
    df_trades = pd.DataFrame([
        {"shares": 100.0, "executable_ask": 0.50, "target": 1},
        {"shares": 100.0, "executable_ask": 0.50, "target": 0},
    ])
    # gross = (100 * 1) - (200 * 0.5) = 100 - 100 = 0.0
    # base fee = 100 * 0.002 = 0.20 -> step0_pnl = -0.20
    # scenario fee (0.001) = 0.10 -> step1_pnl = -0.10
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
    # total shares=30, total cash=15, gross pnl = 5.0
    # baseline fee (0.2%) = 15 * 0.002 = 0.03
    # baseline net pnl = 5.0 - 0.03 = 4.97
    # breakeven fee rate = 5.0 / 15 = 0.3333 (33.33%)
    # breakeven slippage per share = 4.97 / 30 = 0.1657
    be = compute_breakeven_thresholds(df, baseline_fee_rate=0.002)
    assert be["n_trades"] == 3
    assert abs(be["breakeven_fee_rate"] - (5.0 / 15.0)) < 1e-4
    assert abs(be["breakeven_slippage_per_share_usdc"] - (4.97 / 30.0)) < 1e-4

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
