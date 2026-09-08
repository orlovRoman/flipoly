import pytest
import pandas as pd
from polyflip.research.outsider_replay import OutsiderReplayEngine, ReplayPolicy
from polyflip.research.reporting_helpers import generate_price_bins_report

def test_price_boundaries():
    policy = ReplayPolicy(
        max_positions_per_market=10, # allow multiple for test
        stake_usdc=1.0,
        initial_capital=100.0,
        min_edge=0.0, # no edge filter
        default_fee_rate=0.0, # no fees
        max_price=0.40,
    )
    engine = OutsiderReplayEngine(policy=policy)
    
    # Check directly below, exactly at, and above 0.30 and 0.40
    prices = [0.29, 0.30, 0.31, 0.39, 0.40, 0.41, 0.45, 0.50]
    decisions = []
    for i, p in enumerate(prices):
        decisions.append({
            "market_id": f"m_{i}",
            "decision_at": pd.Timestamp("2026-01-01 12:00:00+00:00"),
            "time_left_min": 10.0,
            "candidate_side": "UP",
            "executable_ask": p,
            "p_win": 0.9,
            "target": 1,
        })
    
    ledger = engine.run(decisions)
    trades = ledger.executed_trades
    executed_prices = [t["quote_ask"] for t in trades]
    
    # Should include 0.40, but exclude 0.41, 0.45, 0.50
    assert 0.29 in executed_prices
    assert 0.30 in executed_prices
    assert 0.31 in executed_prices
    assert 0.39 in executed_prices
    assert 0.40 in executed_prices
    assert 0.41 not in executed_prices
    assert 0.45 not in executed_prices
    
    # Test bin [0.40, 0.45)
    df = pd.DataFrame(trades)
    # The trades df has columns: market_id, decision_at, quote_ask, etc.
    df = df.rename(columns={"quote_ask": "executable_ask", "realized_pnl": "pnl"})
    # add outcome
    df["outcome"] = 1
    
    report = generate_price_bins_report(df, bin_width=0.05, min_price=0.25, max_price=0.50)
    # Check if [0.40, 0.45) is not entirely empty because it contains 0.40
    bin_40 = report[report["price_bin"] == "[0.40, 0.45)"].iloc[0]
    assert bin_40["n_rows"] > 0

def test_recalculate_filter_by_trades():
    # 22. Пересчитать фильтр по сделкам, а не суммам корзин
    policy_base = ReplayPolicy(max_price=0.95, min_edge=0.0, default_fee_rate=0.0)
    policy_cap = ReplayPolicy(max_price=0.40, min_edge=0.0, default_fee_rate=0.0)
    
    engine_base = OutsiderReplayEngine(policy=policy_base)
    engine_cap = OutsiderReplayEngine(policy=policy_cap)
    
    prices = [0.20, 0.30, 0.40, 0.45, 0.50]
    decisions = []
    for i, p in enumerate(prices):
        decisions.append({
            "market_id": f"m_{i}",
            "decision_at": pd.Timestamp("2026-01-01 12:00:00+00:00"),
            "time_left_min": 10.0,
            "candidate_side": "UP",
            "executable_ask": p,
            "p_win": 0.9,
            "target": 1,
        })
        
    ledger_base = engine_base.run(decisions)
    ledger_cap = engine_cap.run(decisions)
    
    pnl_base = ledger_base.total_pnl
    pnl_cap = ledger_cap.total_pnl
    
    # Calculate excluded PnL explicitly
    excluded_trades = [t for t in ledger_base.executed_trades if t["quote_ask"] > 0.40]
    pnl_excluded = sum(t["realized_pnl"] for t in excluded_trades)
    
    # Self-check: PnL base = PnL cap + PnL excluded
    assert abs(pnl_base - (pnl_cap + pnl_excluded)) < 1e-3
