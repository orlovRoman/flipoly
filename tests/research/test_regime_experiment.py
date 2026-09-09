"""
tests/research/test_regime_experiment.py

Unit and regression tests for Stage 4 experimental accounting:
- Additive ledger invariant check
- Paired block bootstrap zero self-comparison
- Prevented losses vs missed gains decomposition
"""
import pandas as pd
import numpy as np
import pytest

from polyflip.research.regime_experiment import run_paired_experiment


def test_paired_experiment_invariants_and_self_comparison():
    # Synthetic dataset of 6 opportunities
    df = pd.DataFrame([
        {
            "market_id": "m1",
            "decision_at": "2026-08-10T12:00:00Z",
            "executable_ask": 0.30,
            "target": 1,  # WIN: shares=1/0.3=3.333, pnl=3.333*(1-0.3)-0.002 = +2.331
            "is_reconstructed": False,
            "token_regime": "REVERSION",
            "spot_regime": "REVERSION",
            "reversion_helps_strike": True,
        },
        {
            "market_id": "m2",
            "decision_at": "2026-08-10T12:15:00Z",
            "executable_ask": 0.25,
            "target": 0,  # LOSS: shares=1/0.25=4.0, pnl=4.0*(0-0.25)-0.002 = -1.002
            "is_reconstructed": False,
            "token_regime": "TREND",
            "spot_regime": "TREND",
            "reversion_helps_strike": False,
        },
        {
            "market_id": "m3",
            "decision_at": "2026-08-11T12:00:00Z",
            "executable_ask": 0.35,
            "target": 0,  # LOSS
            "is_reconstructed": False,
            "token_regime": "REVERSION",
            "spot_regime": "QUIET",
            "reversion_helps_strike": False,  # Fails C2 strike context
        },
        {
            "market_id": "m4",
            "decision_at": "2026-08-11T12:15:00Z",
            "executable_ask": 0.20,
            "target": 1,  # WIN
            "is_reconstructed": False,
            "token_regime": "REVERSION",
            "spot_regime": "REVERSION",
            "reversion_helps_strike": True,
        },
        {
            "market_id": "m5",
            "decision_at": "2026-08-12T12:00:00Z",
            "executable_ask": 0.45,  # Exceeds 0.40 -> excluded from C0
            "target": 1,
            "is_reconstructed": False,
            "token_regime": "REVERSION",
            "spot_regime": "REVERSION",
            "reversion_helps_strike": True,
        },
        {
            "market_id": "m6",
            "decision_at": "2026-08-12T12:15:00Z",
            "executable_ask": 0.30,
            "target": 1,
            "is_reconstructed": True,  # Reconstructed quote -> excluded when observed_quotes_only=True
            "token_regime": "REVERSION",
            "spot_regime": "REVERSION",
            "reversion_helps_strike": True,
        },
    ])

    res = run_paired_experiment(df, max_price=0.40, observed_quotes_only=True)
    assert res["dataset_summary"]["c0_opportunities"] == 4  # m1, m2, m3, m4

    # Verify invariants
    inv = res["invariants"]
    assert inv["c1_additive_invariant_holds"] is True
    assert inv["c2_additive_invariant_holds"] is True
    assert inv["self_comparison_zero_delta"] is True

    # Check trade counts
    assert res["variants"]["C0"]["n_trades"] == 4
    # C1 accepts m1, m3, m4 (rejects m2) -> 3 trades
    assert res["variants"]["C1"]["n_trades"] == 3
    # C2 accepts m1, m4 (rejects m3) -> 2 trades
    assert res["variants"]["C2"]["n_trades"] == 2

    # Prevented loss: m2 was a loss (-1.002) prevented by C1
    assert res["disentangled_contributions"]["C1_minus_C0"]["prevented_losses"] > 1.0
    # Missed gain in C1 is 0.0 because m2 was a loss
    assert res["disentangled_contributions"]["C1_minus_C0"]["missed_gains"] == 0.0

    # m3 was a loss prevented by C2
    assert res["disentangled_contributions"]["C2_minus_C1"]["prevented_losses"] > 0.0
