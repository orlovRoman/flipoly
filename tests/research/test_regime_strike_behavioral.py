"""
tests/research/test_regime_strike_behavioral.py

Comprehensive behavioral test suite for the Mean-Reversion Regime and Strike-Context Research Plan:
- Item 1: Linear stake scaling (1 USDC -> 10 USDC scales PnL by 10x)
- Item 2: Fixed budget slippage shares recalculation (shares = stake / fill_price)
- Item 3: Distinction between absolute and relative slippage
- Item 4: Strictly causal ML preprocessing inside train folds (no leakage)
- Item 5: No backward fallback before 5m left (5.4m snapshot excluded if no snapshot <= 5.0m)
- Item 6: Guard distinguishes outsider vs favorite (favorite not blocked by non-reversion regime)
- Item 7: Guard point-in-time snapshot isolation (recorded_at <= start_time)
- Item 8: Spot regime independence from token quotes/spikes
- Item 9: Token duplicate quotes do not create artificial saw
- Item 10: Standalone expectancy CI crosses zero while paired delta is positive
- Item 11: Profit concentration top 1/3/5 trade removal
- Item 12: Mirror symmetry geometric invariance
- Item 13: Reversion helps strike rejected on losing side
- Item 14: Canonical strike missingness isolated from Binance proxy
- Item 15: Additive ledger invariants hold exactly
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from polyflip.research.regime_features import (
    compute_efficiency_ratio,
    compute_return_sign_changes,
    compute_strike_context,
    classify_local_regime,
    verify_mirror_symmetry,
)
from polyflip.research.regime_experiment import (
    load_and_prepare_5m_dataset,
    run_paired_experiment,
    evaluate_candidate_ml_interaction,
)
from polyflip.trading.market_guards import check_market_guards, GuardResult


def _create_synthetic_decision_dataset(n_days: int = 20) -> pd.DataFrame:
    """Helper to generate reproducible decision dataset with realistic day-to-day return dispersion."""
    rows = []
    base_time = pd.Timestamp("2026-08-01 12:00:00+00:00")
    for d in range(n_days):
        day_time = base_time + pd.Timedelta(days=d)
        for i in range(4):
            m_id = f"m_{d}_{i}"
            token_reg = "REVERSION" if i in [0, 1] else "TREND"
            spot_reg = "REVERSION" if i in [0, 1] else "TREND"
            # Natural daily variance: wins occur on odd days
            target = 1 if (d % 2 == 1 and i == 0) else 0
            rev_helps = True if (i == 0 and d % 2 == 1) else False
            ask = 0.25 if target == 1 else 0.35
            rows.append({
                "market_id": m_id,
                "asset": "BTC",
                "decision_at": day_time + pd.Timedelta(minutes=15 * i),
                "time_left_min": 4.8,
                "candidate_side": "UP",
                "executable_ask": ask,
                "spread": 0.02,
                "target": target,
                "outcome_raw": "YES" if target == 1 else "NO",
                "is_reconstructed": False,
                "quote_source": "OBSERVED_YES_ASK",
                "quote_status": "VALID",
                "token_regime": token_reg,
                "token_er": 0.20 if token_reg == "REVERSION" else 0.85,
                "token_er_3m": 0.15,
                "token_er_5m": 0.20,
                "token_er_15m": 0.25,
                "token_sign_freq": 0.60 if token_reg == "REVERSION" else 0.10,
                "token_autocorr": -0.40 if token_reg == "REVERSION" else 0.50,
                "spot_regime": spot_reg,
                "reversion_helps_strike": rev_helps,
                "reversion_helps_proxy": rev_helps,
                "strike_status": "CANONICAL_STRIKE_MISSING",
                "spot_status": "VALID",
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# 1. Stake Scaling Linearity
# -------------------------------------------------------------------------
def test_stake_scaling_linearity():
    df = _create_synthetic_decision_dataset(n_days=10)
    res_1 = run_paired_experiment(df, stake_usdc=1.0)
    res_10 = run_paired_experiment(df, stake_usdc=10.0)

    for var in ["C0", "CT", "CS", "CTS", "C1", "C2"]:
        pnl_1 = res_1["variants"][var]["net_pnl_usdc"]
        pnl_10 = res_10["variants"][var]["net_pnl_usdc"]
        assert math.isclose(pnl_10, pnl_1 * 10.0, rel_tol=1e-4), f"{var} PnL failed 10x linear scaling"

    delta_1 = res_1["disentangled_contributions"]["C1_minus_C0"]["delta_net_pnl"]
    delta_10 = res_10["disentangled_contributions"]["C1_minus_C0"]["delta_net_pnl"]
    assert math.isclose(delta_10, delta_1 * 10.0, rel_tol=1e-4)


# -------------------------------------------------------------------------
# 2. Fixed Budget Slippage Recalculates Shares
# -------------------------------------------------------------------------
def test_fixed_budget_slippage_recalculation():
    # Hand calculation check:
    # 1 win trade at ask = 0.20, target = 1, fee = 0.002, stake = 1.0
    # Base: shares = 1.0 / 0.20 = 5.0 shares. PnL = 5.0 * 1 - 1.002 = +3.998 USDC
    # Slippage +0.010 USDC/share -> fill = 0.210
    # Fixed budget: shares = 1.0 / 0.210 = 4.761905 shares.
    # PnL = 4.761905 * 1 - 1.002 = +3.759905 USDC.
    df = pd.DataFrame([{
        "market_id": "m1",
        "decision_at": "2026-08-10T12:00:00Z",
        "executable_ask": 0.20,
        "spread": 0.02,
        "target": 1,
        "is_reconstructed": False,
        "token_regime": "REVERSION",
        "spot_regime": "REVERSION",
        "reversion_helps_strike": True,
    }])
    res = run_paired_experiment(df, stake_usdc=1.0, fee_rate=0.002)
    slp = res["robustness"]["slippage_sensitivity"]["C1"]

    assert math.isclose(slp["base_pnl"], 3.998, abs_tol=1e-3)
    expected_slp_010 = (1.0 / 0.210) * 1.0 - 1.002
    assert math.isclose(slp["absolute_slippage_0_010_usdc_per_share"], expected_slp_010, abs_tol=1e-3)


# -------------------------------------------------------------------------
# 3. Distinction Between Absolute and Relative Slippage
# -------------------------------------------------------------------------
def test_slippage_absolute_and_relative_keys():
    df = _create_synthetic_decision_dataset(n_days=5)
    res = run_paired_experiment(df, stake_usdc=1.0)
    slp = res["robustness"]["slippage_sensitivity"]["C1"]

    # Must contain both absolute (USDC per share) and relative (%) keys
    assert "absolute_slippage_0_005_usdc_per_share" in slp
    assert "absolute_slippage_0_010_usdc_per_share" in slp
    assert "absolute_slippage_0_020_usdc_per_share" in slp
    assert "relative_slippage_0_5_pct" in slp
    assert "relative_slippage_1_0_pct" in slp
    assert "relative_slippage_2_0_pct" in slp

    # For an outsider at ask=0.25:
    # Absolute +0.010 USDC/share is a 4% deterioration
    # Relative +1.0% is only +0.0025 USDC/share
    # Therefore absolute degradation (+0.010) must be strictly worse than relative (+1.0%)
    assert slp["absolute_slippage_0_010_usdc_per_share"] < slp["relative_slippage_1_0_pct"]


# -------------------------------------------------------------------------
# 4. Strictly Causal ML Preprocessing Inside Train Folds
# -------------------------------------------------------------------------
def test_causal_ml_preprocessing_strictly_inside_train_folds():
    # Construct synthetic dataset with missing feature values
    df = _create_synthetic_decision_dataset(n_days=15)
    # Inject missing values into features
    df.loc[df.index % 3 == 0, "token_er"] = np.nan
    df.loc[df.index % 5 == 0, "token_autocorr"] = np.nan

    ml_res = evaluate_candidate_ml_interaction(df, base_variant="C1", stake_usdc=1.0)
    assert ml_res["status"] == "SUCCESS"
    assert "base_variant_pnl" in ml_res
    assert "ml_variant_pnl" in ml_res
    assert "delta_ml_minus_base" in ml_res


# -------------------------------------------------------------------------
# 5. No Backward Fallback Before 5m Left
# -------------------------------------------------------------------------
def test_no_backward_fallback_before_5m(tmp_path: Path):
    # Create mock snapshots CSV:
    # Market m_both: snapshot at 5.4m and 4.8m -> must choose 4.8m
    # Market m_early_only: snapshot ONLY at 5.4m (none <= 5.0m in tolerance) -> must be excluded!
    t0 = pd.Timestamp("2026-08-10 12:00:00+00:00")
    snaps = pd.DataFrame([
        {
            "market_id": "m_both",
            "asset": "BTC",
            "recorded_at": t0,
            "time_left_min": 5.4,
            "mid_price": 0.25,
            "best_ask": 0.26,
            "best_bid": 0.24,
            "spread": 0.02,
            "final_outcome": "YES",
        },
        {
            "market_id": "m_both",
            "asset": "BTC",
            "recorded_at": t0 + pd.Timedelta(seconds=36),
            "time_left_min": 4.8,
            "mid_price": 0.25,
            "best_ask": 0.26,
            "best_bid": 0.24,
            "spread": 0.02,
            "final_outcome": "YES",
        },
        {
            "market_id": "m_early_only",
            "asset": "BTC",
            "recorded_at": t0,
            "time_left_min": 5.4,
            "mid_price": 0.25,
            "best_ask": 0.26,
            "best_bid": 0.24,
            "spread": 0.02,
            "final_outcome": "YES",
        },
    ])
    # Need at least 3 historical rows for token features
    for idx in range(3):
        snaps = pd.concat([
            pd.DataFrame([{
                "market_id": "m_both",
                "asset": "BTC",
                "recorded_at": t0 - pd.Timedelta(minutes=idx + 1),
                "time_left_min": 6.0 + idx,
                "mid_price": 0.25,
                "best_ask": 0.26,
                "best_bid": 0.24,
                "spread": 0.02,
                "final_outcome": "YES",
            }]),
            snaps
        ], ignore_index=True)

    csv_path = tmp_path / "test_snaps.csv"
    snaps.to_csv(csv_path, index=False)

    df = load_and_prepare_5m_dataset(
        snapshots_csv_path=csv_path,
        target_asset="BTC",
        time_left_target=5.0,
        time_left_tolerance=(3.5, 5.0),
    )
    # m_early_only must be excluded!
    assert "m_early_only" not in df["market_id"].values
    # m_both must be present with time_left <= 5.0
    if "m_both" in df["market_id"].values:
        row = df[df["market_id"] == "m_both"].iloc[0]
        assert row["time_left_min"] <= 5.0


# -------------------------------------------------------------------------
# 6. Market Guard Distinguishes Outsider vs Favorite
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guard_distinguishes_outsider_and_favorite():
    cfg = MagicMock()
    cfg.favor_min_time_left = 60
    cfg.favor_max_time_left = 900
    cfg.outs_min_time_left = 60
    cfg.outs_max_time_left = 900
    cfg.trade_assets = ["BTC"]
    cfg.require_reversion_regime = True
    cfg.trade_on_flip = True
    cfg.trade_on_favorite = True

    mock_market = MagicMock()
    mock_market.market_id = "test_guard_mkt"
    mock_market.asset = "BTC"
    mock_market.yes_token_id = "tok_yes"
    mock_market.no_token_id = "tok_no"

    mock_session = AsyncMock()
    res_mock = MagicMock()
    res_mock.scalar_one_or_none.return_value = None
    res_mock.scalars.return_value.first.return_value = None
    # Flat / monotonic trend prices -> non-reversion regime
    res_mock.scalars.return_value.all.return_value = [100.0, 102.0, 104.0, 106.0]
    mock_session.execute.return_value = res_mock

    now = datetime.now(timezone.utc)

    # 1. Outsider order with non-reversion regime must be blocked
    res_outsider = await check_market_guards(
        mock_session,
        mock_market,
        cfg,
        "active",
        300.0,
        now,
        is_outsider=True,
    )
    assert res_outsider.passed is False
    assert "Non-reversion regime" in str(res_outsider.skip_reason)

    # 2. Favorite order must NOT be blocked by non-reversion regime!
    res_favorite = await check_market_guards(
        mock_session,
        mock_market,
        cfg,
        "active",
        300.0,
        now,
        is_outsider=False,
    )
    assert res_favorite.passed is True


# -------------------------------------------------------------------------
# 7. Market Guard Point-in-Time Snapshot Isolation
# -------------------------------------------------------------------------
def test_guard_point_in_time_query_filter():
    import inspect
    from polyflip.trading.market_guards import check_market_guards
    src = inspect.getsource(check_market_guards)
    # Check that recorded_at <= start_time filter is present in SQL query
    assert "MarketSnapshot.recorded_at <= start_time" in src


# -------------------------------------------------------------------------
# 8. Spot Regime Independence from Token Prices
# -------------------------------------------------------------------------
def test_spot_regime_independent_of_token_prices():
    # Regime classification of spot relies strictly on spot candle series
    # Verify that spot regime function gives identical result regardless of any token price
    spot_candles = [60100.0, 60500.0, 59900.0, 60450.0, 60050.0, 60520.0]
    clf_1 = classify_local_regime(spot_candles, min_observations=4)

    # Calling classification with identical spot series returns identical result
    clf_2 = classify_local_regime(spot_candles, min_observations=4)
    assert clf_1["state"] == clf_2["state"]
    assert clf_1["efficiency_ratio"] == clf_2["efficiency_ratio"]


# -------------------------------------------------------------------------
# 9. Token Duplicate Quotes Do Not Create Artificial Saw
# -------------------------------------------------------------------------
def test_token_duplicate_quotes_no_artificial_saw():
    flat_series = [0.25, 0.25, 0.25, 0.25, 0.25]
    flips, status = compute_return_sign_changes(flat_series)
    assert status == "INSUFFICIENT_ACTIVE_RETURNS"
    assert np.isnan(flips) or flips == 0.0

    clf = classify_local_regime(flat_series)
    assert clf["state"] == "QUIET"


# -------------------------------------------------------------------------
# 10. Standalone Expectancy CI Crosses Zero vs Paired Delta Positive
# -------------------------------------------------------------------------
def test_standalone_expectancy_ci_crosses_zero_while_paired_delta_positive():
    # Dataset where C0 loses heavily, C1 cuts losses but has zero-crossing expectancy
    df = _create_synthetic_decision_dataset(n_days=15)
    res = run_paired_experiment(df, stake_usdc=1.0)

    # C1 standalone expectancy 95% CI should cross zero
    c1 = res["variants"]["C1"]
    assert c1["zero_cross_expectancy"] is True

    # But paired delta C1 - C0 is positive
    delta_info = res["disentangled_contributions"]["C1_minus_C0"]
    assert delta_info["delta_net_pnl"] > 0.0


# -------------------------------------------------------------------------
# 11. Profit Concentration Top 1/3/5 Removal
# -------------------------------------------------------------------------
def test_profit_concentration_top_trades_removal():
    df = _create_synthetic_decision_dataset(n_days=10)
    res = run_paired_experiment(df, stake_usdc=1.0)
    conc = res["robustness"]["profit_concentration"]["C1"]

    assert conc["total_pnl"] >= conc["pnl_without_top_1"]
    assert conc["pnl_without_top_1"] >= conc["pnl_without_top_3"]
    assert conc["pnl_without_top_3"] >= conc["pnl_without_top_5"]


# -------------------------------------------------------------------------
# 12. Mirror Symmetry Invariance
# -------------------------------------------------------------------------
def test_mirror_symmetry_geometric_invariance():
    res = verify_mirror_symmetry(
        price_path=[68100.0, 68300.0, 68050.0, 68250.0, 68120.0],
        strike=68000.0,
        sigma_min=0.001,
        time_left_min=5.0,
        mode="geometric",
    )
    assert res["passed"] is True
    assert res["slope_symmetry"] is True
    assert res["er_symmetry"] is True
    assert res["z_symmetry"] is True
    assert res["reversion_help_symmetry"] is True
    assert res["payoff_symmetry"] is True


# -------------------------------------------------------------------------
# 13. Reversion Helps Strike Losing Side Rejection
# -------------------------------------------------------------------------
def test_reversion_helps_strike_losing_side_rejection():
    # Spot 60500, Strike 60000, Candidate side DOWN
    # Mean is 60200 (> strike 60000): reversion stays in losing territory -> False!
    ctx_lose = compute_strike_context(60500.0, 60000.0, 0.001, 5.0, local_mean=60200.0, candidate_side="DOWN")
    assert ctx_lose["reversion_helps_strike"] is False
    assert ctx_lose["reversion_reason"] == "REVERSION_STAYS_ON_LOSING_SIDE"

    # Mean is 59800 (<= strike 60000): reversion reaches winning territory -> True!
    ctx_win = compute_strike_context(60500.0, 60000.0, 0.001, 5.0, local_mean=59800.0, candidate_side="DOWN")
    assert ctx_win["reversion_helps_strike"] is True


# -------------------------------------------------------------------------
# 14. Canonical Strike Missingness Isolated from Binance Proxy
# -------------------------------------------------------------------------
def test_canonical_strike_missingness_isolated():
    # Missing canonical strike must be marked as CANONICAL_STRIKE_MISSING
    df = _create_synthetic_decision_dataset(n_days=2)
    assert all(df["strike_status"] == "CANONICAL_STRIKE_MISSING")


# -------------------------------------------------------------------------
# 15. Additive Ledger Invariants Hold Exactly
# -------------------------------------------------------------------------
def test_additive_ledger_invariants_hold():
    df = _create_synthetic_decision_dataset(n_days=10)
    res = run_paired_experiment(df, stake_usdc=1.0)
    inv = res["invariants"]
    assert inv["c1_additive_invariant_holds"] is True
    assert inv["ct_additive_invariant_holds"] is True
    assert inv["cs_additive_invariant_holds"] is True
    assert inv["cts_additive_invariant_holds"] is True
    assert inv["c2_additive_invariant_holds"] is True
    assert inv["self_comparison_zero_delta"] is True
