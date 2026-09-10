"""
tests/research/outsider_price_path/test_evaluation.py

Unit tests for evaluation metrics, day-block bootstrap, and stratification (Stages 4, 5):
- Summary statistics computation
- Stratification matching and coverage loss accounting
- Day-block paired bootstrap and CI calculation
- Holm-Bonferroni multiple testing adjustment
- Concentration and stress test audit
"""
import numpy as np
import pandas as pd
import pytest

from polyflip.research.outsider_price_path.evaluation import (
    summarize_policy,
    compute_stratified_comparison,
    run_day_block_bootstrap,
    compute_concentration_audit,
)


def _make_sample_df() -> pd.DataFrame:
    records = []
    # 20 days, 10 markets per day = 200 markets
    for day_i in range(20):
        cal_date = f"2026-08-{day_i+1:02d}"
        week = f"2026-W{31 + (day_i // 7):02d}"
        for m_i in range(10):
            cohort = "FORMER_FAVORITE" if m_i < 4 else "OBSERVED_ALWAYS_OUTSIDER"
            rebound = "REBOUND" if m_i in (2, 3, 6, 7) else "NO_REBOUND"
            ask = 0.20 if cohort == "FORMER_FAVORITE" else 0.25
            target = 1 if (m_i % 3 == 0) else 0
            gross = (1.0 / ask) * (target - ask)
            net_02 = gross - 0.002
            records.append({
                "market_id": f"{day_i}_{m_i}",
                "asset": "BTC" if m_i % 2 == 0 else "ETH",
                "side": "UP",
                "calendar_date": cal_date,
                "calendar_week": week,
                "ask": ask,
                "ask_bin": "[0.20,0.25)" if ask == 0.20 else "[0.25,0.30)",
                "target": target,
                "budget_usdc": 1.0,
                "shares": 1.0 / ask,
                "gross_pnl": gross,
                "net_pnl_02pct": net_02,
                "net_pnl_01pct": gross - 0.001,
                "primary_cohort": cohort,
                "rebound_category": rebound,
                "ct_state": "REVERSION" if m_i in (1, 2) else "TREND",
            })
    return pd.DataFrame(records)


def test_summarize_policy():
    df = _make_sample_df()
    summ = summarize_policy(df, "All_Markets", total_universe_trades=400)
    assert summ.n_trades == 200
    assert summ.n_days == 20
    assert 0.0 <= summ.win_rate <= 1.0
    assert summ.turnover_usdc == 200.0
    assert summ.expectancy_usdc == summ.net_pnl_02pct / 200
    assert summ.total_universe_trades == 400
    assert abs(summ.stream_expectancy_usdc - (summ.net_pnl_02pct / 400)) < 1e-6
    assert summ.filter_pass_rate_pct == 50.0


def test_stratified_comparison():
    df = _make_sample_df()
    res = compute_stratified_comparison(df, "primary_cohort")
    assert "common_strata_count" in res
    assert "coverage_loss_pct" in res
    assert "adjusted_win_rate_diff" in res
    assert "adjusted_expectancy_diff_usdc" in res
    assert "equal_strata_win_rate_diff" in res
    assert "equal_strata_expectancy_diff_usdc" in res


def test_day_block_bootstrap():
    df = _make_sample_df()
    policies = {
        "Control": pd.Series([True] * len(df)),
        "Former_Favorite": df["primary_cohort"] == "FORMER_FAVORITE",
        "Rebound": df["rebound_category"] == "REBOUND",
    }
    boot = run_day_block_bootstrap(df, policies, n_replicates=100, seed=123)
    assert "policies" in boot
    assert "paired_differences" in boot
    assert "Former_Favorite_minus_Control" in boot["paired_differences"]
    diff_res = boot["paired_differences"]["Former_Favorite_minus_Control"]
    assert "d_expectancy_ci95" in diff_res
    assert "holm_adj_p_value" in diff_res


def test_concentration_audit():
    df = _make_sample_df()
    audit = compute_concentration_audit(df)
    assert "top5_wins_sum" in audit
    assert "best_day" in audit
    assert "weekly_summary" in audit
    assert audit["weekly_summary"]["total_weeks"] > 0
    assert "by_side" in audit
    assert len(audit["by_side"]) > 0


def test_stratified_comparison_with_overlap():
    records = []
    for day_i in range(10):
        cal_date = f"2026-08-{day_i+1:02d}"
        week = f"2026-W32"
        for m_i in range(10):
            cohort = "FORMER_FAVORITE" if m_i % 2 == 0 else "OBSERVED_ALWAYS_OUTSIDER"
            ask = 0.12 if cohort == "FORMER_FAVORITE" else 0.11
            target = 1 if (m_i % 3 == 0) else 0
            gross = (1.0 / ask) * (target - ask)
            records.append({
                "market_id": f"{day_i}_{m_i}",
                "asset": "BTC",
                "side": "UP",
                "calendar_date": cal_date,
                "calendar_week": week,
                "ask": ask,
                "ask_bin": "[0.10,0.15)",
                "target": target,
                "budget_usdc": 1.0,
                "net_pnl_02pct": gross - 0.002,
                "primary_cohort": cohort,
            })
    df = pd.DataFrame(records)
    res = compute_stratified_comparison(df, "primary_cohort", n_boot=50, seed=42)
    assert res["common_strata_count"] > 0
    assert res["coverage_loss_pct"] == 0.0
    assert "adjusted_win_rate_diff_ci95" in res
    assert "residual_ask_diff_precision" in res
    assert abs(res["residual_ask_diff_precision"] - 0.01) < 1e-4


def test_day_block_bootstrap_point_estimates_and_stream():
    df = _make_sample_df()
    policies = {
        "Control": pd.Series([True] * len(df)),
        "Former_Favorite": df["primary_cohort"] == "FORMER_FAVORITE",
    }
    boot = run_day_block_bootstrap(df, policies, n_replicates=50, seed=42)
    # Check own policy stats
    assert "point_estimate" in boot["policies"]["Control"]
    assert "p_value_own_profitability" in boot["policies"]["Control"]
    assert "stream_expectancy_mean" in boot["policies"]["Former_Favorite"]
    # Check paired difference stats
    pair = boot["paired_differences"]["Former_Favorite_minus_Control"]
    assert "point_estimate" in pair
    assert "d_trade_expectancy_mean" in pair
    assert "d_stream_expectancy_mean" in pair
    assert "trade_exp_nominal_p_value" in pair
    assert "stream_exp_nominal_p_value" in pair
    assert "stream_exp_is_significant_05" in pair

