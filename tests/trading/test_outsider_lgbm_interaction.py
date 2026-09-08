"""
tests/trading/test_outsider_lgbm_interaction.py

Unit tests for Item 2.9: LightGBM interaction with Outsider Model B
(standalone B, B + LGBM veto, B + LGBM input meta-model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polyflip.trading.combined_voting import evaluate_lgbm_outsider_interaction


@pytest.fixture
def sample_interaction_dataset() -> pd.DataFrame:
    """Generates synthetic dataset with Model B and LightGBM outputs."""
    np.random.seed(42)
    n = 40
    rows = []
    for i in range(n):
        side = "UP" if i % 2 == 0 else "DOWN"
        # Model B probabilities: high edge for several trades
        p_b = 0.55 if i < 20 else 0.35
        ask = 0.30  # EV > 0.02 for p_b=0.55 or p_b=0.35
        target = int(np.random.rand() < (0.6 if side == "UP" else 0.4))

        # LightGBM opening direction
        # Half agree, half disagree
        l_dir = side if i % 4 in (0, 1) else ("DOWN" if side == "UP" else "UP")
        l_prob = 0.65 if l_dir == "UP" else 0.35

        rows.append({
            "market_id": f"m_{i // 2}",
            "candidate_side": side,
            "p_b_win": p_b,
            "executable_ask": ask,
            "target": target,
            "lgbm_direction": l_dir,
            "lgbm_oof_prob": l_prob,
        })
    return pd.DataFrame(rows)


def test_veto_partition_invariant(sample_interaction_dataset: pd.DataFrame):
    """Item 2.9: n_accepted + n_vetoed == n_total_candidates."""
    res = evaluate_lgbm_outsider_interaction(sample_interaction_dataset, min_edge=0.02)

    n_candidates = res["n_total_candidates"]
    assert n_candidates > 0

    veto_res = res["b_plus_lgbm_veto"]
    n_acc = veto_res["n_accepted"]
    n_vet = veto_res["n_vetoed"]

    assert n_acc + n_vet == n_candidates


def test_veto_counterfactual_accounting(sample_interaction_dataset: pd.DataFrame):
    """Item 2.9: Every vetoed trade has counterfactual outcome and pnl tracked."""
    res = evaluate_lgbm_outsider_interaction(sample_interaction_dataset, min_edge=0.02)
    veto_res = res["b_plus_lgbm_veto"]
    cf = veto_res["counterfactual"]

    assert cf["n_vetoed"] == veto_res["n_vetoed"]
    assert 0 <= cf["vetoed_wins"] <= cf["n_vetoed"]
    assert "saved_losses_pnl" in cf
    assert "missed_gains_pnl" in cf
    assert "vetoed_pnl_counterfactual" in cf


def test_meta_model_input_combines_signals(sample_interaction_dataset: pd.DataFrame):
    """Item 2.9: Meta-model trains on Model B and OOF LGBM probabilities."""
    res = evaluate_lgbm_outsider_interaction(sample_interaction_dataset, min_edge=0.02)
    meta_res = res["b_plus_lgbm_input"]

    assert "meta_weights" in meta_res
    weights = meta_res["meta_weights"]
    assert "w_model_b" in weights
    assert "w_lgbm" in weights
    assert meta_res["n_trades"] >= 0
