"""
tests/models/test_outsider_baselines.py

Unit tests for transparent price baselines (M0 and Mlegacy, Item 2.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polyflip.models.outsider_baselines import (
    MarketPriceBaseline,
    LegacyOutsiderBaseline,
    evaluate_outsider_predictions,
)


def test_m0_market_price_baseline_exact_arithmetic():
    """Item 2.5: M0 must reproduce p_win == outsider_mid via pure arithmetic."""
    m0 = MarketPriceBaseline()
    df = pd.DataFrame({
        "outsider_mid": [0.15, 0.30, 0.45, 0.50],
    })
    probs = m0.predict_proba(df)

    assert probs.shape == (4, 2)
    # Check p_win (column 1) matches outsider_mid
    assert probs[:, 1] == pytest.approx(np.array([0.15, 0.30, 0.45, 0.50]), abs=1e-6)
    # Check sum of binary probabilities equals 1
    assert (probs[:, 0] + probs[:, 1]) == pytest.approx(1.0, abs=1e-9)


def test_mlegacy_baseline_consistency():
    """Item 2.5: Mlegacy incorporates velocity and spread penalties while preserving bounds."""
    m_legacy = LegacyOutsiderBaseline(velocity_weight=0.1, spread_penalty=0.5)
    df = pd.DataFrame({
        "outsider_mid": [0.40, 0.40],
        "price_velocity": [0.05, -0.05],
        "spread": [0.02, 0.04],
    })
    probs = m_legacy.predict_proba(df)

    p1 = 0.40 + 0.1 * 0.05 - 0.5 * 0.02  # 0.40 + 0.005 - 0.01 = 0.395
    p2 = 0.40 + 0.1 * (-0.05) - 0.5 * 0.04  # 0.40 - 0.005 - 0.02 = 0.375

    assert probs[0, 1] == pytest.approx(p1, rel=1e-5)
    assert probs[1, 1] == pytest.approx(p2, rel=1e-5)


def test_evaluate_outsider_predictions_metrics():
    """Item 2.5: Evaluates canonical metrics and economic simulation."""
    y_true = np.array([1, 0, 1, 0])
    p_win = np.array([0.40, 0.30, 0.60, 0.20])
    executable_ask = np.array([0.35, 0.35, 0.45, 0.30])

    eval_res = evaluate_outsider_predictions(
        y_true=y_true,
        p_win=p_win,
        executable_ask=executable_ask,
        fee_rate=0.002,
        min_edge=0.02,
    )

    assert eval_res["n_samples"] == 4
    assert 0.0 <= eval_res["brier"] <= 1.0
    assert eval_res["log_loss"] > 0.0
    assert 0.0 <= eval_res["ece"] <= 1.0
    assert eval_res["n_trades"] >= 0
