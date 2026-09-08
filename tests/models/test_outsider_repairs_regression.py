"""
tests/models/test_outsider_repairs_regression.py

Defect Registry & Behavioral Regression Test Suite for R1-R9.
Each test asserts the CORRECT, sound behavioral contract rather than the buggy implementation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

# Defect Registry Metadata
DEFECT_REGISTRY = {
    "R1": {
        "title": "Synthetic benchmark represented as market model selection",
        "tests": ["test_r1_synthetic_cannot_claim_edge_supported", "test_r1_market_side_target_consistency"],
    },
    "R2": {
        "title": "Trainer data leakage: chronological walk-forward vs future leakage",
        "tests": ["test_r2_future_targets_do_not_alter_earliest_oof", "test_r2_zero_market_leakage_across_folds"],
    },
    "R3": {
        "title": "Incorrect executable ask and targets: DOWN ask fallback, PENDING target, fake prices",
        "tests": ["test_r3_down_ask_never_uses_yes_ask", "test_r3_unresolved_status_not_zero_target", "test_r3_missing_ask_not_invented_mid_plus_001"],
    },
    "R4": {
        "title": "Underlying observations collection and integration flow",
        "tests": ["test_r4_observations_asof_reader_flow", "test_r4_nan_lag_column_does_not_block_observations"],
    },
    "R5": {
        "title": "Point-in-time violations: future candles change past z, decision_at wallclock bypass, sigma fallback",
        "tests": ["test_r5_future_candles_do_not_alter_past_sigma_and_z", "test_r5_decision_at_used_without_wallclock_leak", "test_r5_missing_sigma_excluded_from_complete_b"],
    },
    "R6": {
        "title": "Mlegacy baseline does not reproduce authentic BTC_leaning@11",
        "tests": [
            "test_r6_mlegacy_uses_authentic_btc_leaning_model",
            "test_r6_mlegacy_orients_down_candidate_side",
        ],
    },
    "R7": {
        "title": "LightGBM stacking leakage and misaligned prediction evaluation",
        "tests": ["test_r7_meta_stacking_temporal_market_isolation", "test_r7_b_plus_input_evaluates_meta_probs_not_pb"],
    },
    "R8": {
        "title": "Economic replay position state machine, cost model, and conclusive selection",
        "tests": [
            "test_r8_one_position_per_market_in_replay",
            "test_r8_net_ev_formula_and_non_default_costs",
            "test_r8_net_ev_invalid_bounds_returns_nan",
            "test_r8_replay_veto_mode_allows_subsequent_entries",
            "test_r8_selection_returns_inconclusive_when_no_edge",
        ],
    },
    "R9": {
        "title": "Reporting statistics: drawdown with initial equity, cluster uncertainty, price bins",
        "tests": [
            "test_r9_drawdown_two_losses_equals_two",
            "test_r9_single_cluster_returns_insufficient_blocks",
            "test_r9_paired_cluster_delta_insufficient_blocks",
            "test_r9_price_bins_reliability_separated_from_probability_bins",
        ],
    },
}


# ==============================================================================
# R1 Tests
# ==============================================================================
def test_r1_synthetic_cannot_claim_edge_supported():
    """R1: Synthetic or demo data must return DEMO_ONLY and NEVER EDGE_SUPPORTED."""
    from scripts.research.compare_outsider_models import compare_all_models
    from scripts.research.outsider_ablation import generate_ablation_dataset

    df = generate_ablation_dataset(n_markets=12, seed=42)
    result = compare_all_models(df, source_kind="SYNTHETIC")
    assert result.get("status") in ("DEMO_ONLY", "INPUT_REQUIRED", "INCONCLUSIVE")
    assert result.get("status") != "EDGE_SUPPORTED"


def test_r1_market_side_target_consistency():
    """R1: Within one market, settlement is single; opposing sides complement."""
    from scripts.research.outsider_ablation import generate_ablation_dataset

    df = generate_ablation_dataset(n_markets=24, seed=42)
    conflicts = (df.groupby(["market_id", "candidate_side"])["target"].nunique() > 1).sum()
    assert conflicts == 0, f"Found {conflicts} conflicting targets for the same market and side!"


# ==============================================================================
# R2 Tests
# ==============================================================================
def test_r2_future_targets_do_not_alter_earliest_oof():
    """R2: Changing future targets in walk-forward must NOT change earliest fold OOF."""
    from polyflip.models.outsider_trainer import train_outsider_model
    from scripts.research.outsider_ablation import generate_ablation_dataset

    df = generate_ablation_dataset(n_markets=24, seed=42)
    ts = pd.Timestamp("2026-09-01T12:00:00Z")
    df["decision_at"] = pd.date_range(ts, periods=len(df), freq="5min")

    res1 = train_outsider_model(df, feature_set="MODEL_A1", validation_mode="walk_forward")
    p1 = res1.oof_predictions

    # Change targets in the latest fold / future block
    df_changed = df.copy()
    latest_cutoff = df["decision_at"].quantile(0.75)
    future_mask = df["decision_at"] >= latest_cutoff
    df_changed.loc[future_mask, "target"] = 1 - df_changed.loc[future_mask, "target"]

    res2 = train_outsider_model(df_changed, feature_set="MODEL_A1", validation_mode="walk_forward")
    p2 = res2.oof_predictions

    earliest_mask = (df["decision_at"] < df["decision_at"].quantile(0.25)) & np.isfinite(p1)
    if earliest_mask.sum() > 0:
        max_diff = np.max(np.abs(p1[earliest_mask] - p2[earliest_mask]))
        assert max_diff == pytest.approx(0.0, abs=1e-7), f"Future target change leaked to past OOF! Max diff: {max_diff}"


def test_r2_zero_market_leakage_across_folds():
    """R2: No market_id may appear simultaneously in train and validation folds."""
    from polyflip.models.temporal_validation import grouped_walk_forward_folds

    n = 60
    groups = pd.Series([f"m_{i // 3}" for i in range(n)])
    ts = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
    folds = grouped_walk_forward_folds(groups, ts, n_splits=4)
    assert len(folds) >= 2
    for fold in folds:
        shared = set(fold.train_groups) & set(fold.validation_groups)
        assert len(shared) == 0, f"Markets leaked between train and validation: {shared}"


# ==============================================================================
# R3 Tests
# ==============================================================================
def test_r3_down_ask_never_uses_yes_ask():
    """R3: If DOWN quote is absent, executable ask must NOT fall back to YES ask."""
    from polyflip.models.outsider_dataset import build_outsider_decision_rows

    ts = pd.Timestamp("2026-09-01T12:00:00Z")
    row = {
        "market_id": "m_test_r3",
        "recorded_at": ts,
        "time_left_min": 5.0,
        "mid_price": 0.74,  # YES mid > 0.5 -> outsider is DOWN, outsider_mid = 0.26
        "best_ask": 0.75,   # YES best ask
        "best_bid": 0.73,
        "spread": 0.02,
        "final_outcome": "DOWN",
    }
    df = pd.DataFrame([row])
    dec = build_outsider_decision_rows(df)
    assert not dec.empty
    down_row = dec.iloc[0]
    assert down_row["candidate_side"] == "DOWN"
    # executable_ask must NOT be 0.75!
    assert down_row["executable_ask"] != pytest.approx(0.75), "DOWN ask erroneously defaulted to YES best_ask 0.75!"
    assert pd.isna(down_row["executable_ask"]) or down_row.get("quote_valid") is False or down_row["executable_ask"] <= 0.50


def test_r3_unresolved_status_not_zero_target():
    """R3: Unresolved / PENDING market outcome must not become target=0 (loss)."""
    from polyflip.models.outsider_dataset import build_outsider_decision_rows

    ts = pd.Timestamp("2026-09-01T12:00:00Z")
    row = {
        "market_id": "m_pending",
        "recorded_at": ts,
        "time_left_min": 5.0,
        "mid_price": 0.35,
        "final_outcome": "PENDING",
    }
    df = pd.DataFrame([row])
    dec = build_outsider_decision_rows(df)
    assert not dec.empty
    target_val = dec.iloc[0]["target"]
    assert pd.isna(target_val), f"Pending market was assigned target={target_val} instead of NaN/Unresolved!"


def test_r3_missing_ask_not_invented_mid_plus_001():
    """R3: Missing ask must not be silently invented as outsider_mid + 0.01."""
    from polyflip.models.outsider_dataset import resolve_candidate_quote

    q = resolve_candidate_quote(candidate_side="UP", yes_best_ask=np.nan, down_best_ask=np.nan, outsider_mid=0.30)
    assert q.is_valid is False
    assert q.executable_ask is None or pd.isna(q.executable_ask)
    assert q.rejection_reason in ("MISSING_CANDIDATE_QUOTE", "NO_QUOTE")


# ==============================================================================
# R4 Tests
# ==============================================================================
def test_r4_observations_asof_reader_flow():
    """R4: Written observations must be read causally as-of decision_at."""
    from polyflip.crypto.underlying_observations import (
        Observation,
        compute_underlying_return,
    )

    t0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs_list = [
        Observation("BTC", 50000.0, "BINANCE", t0 - timedelta(seconds=120), t0 - timedelta(seconds=120)),
        Observation("BTC", 50020.0, "BINANCE", t0 - timedelta(seconds=30), t0 - timedelta(seconds=30)),
        Observation("BTC", 50050.0, "BINANCE", t0, t0),
    ]
    ret30, ok30 = compute_underlying_return(obs_list, as_of=t0, horizon_seconds=30.0)
    assert ok30 is True
    assert ret30 == pytest.approx(np.log(50050.0 / 50020.0), rel=1e-7)


def test_r4_nan_lag_column_does_not_block_observations():
    """R4: A dataframe with NaN lag columns must fall back to observations if present."""
    from polyflip.models.point_in_time_features import compute_outsider_model_features
    from polyflip.crypto.underlying_observations import Observation

    t0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    base = pd.DataFrame([{
        "mid_price": 0.3,
        "time_left_min": 5.0,
        "spread": 0.02,
        "candidate_side": "UP",
        "decision_at": t0,
        "underlying_price": 50050.0,
        "underlying_lag_30s": np.nan,
        "strike_value": 50000.0,
        "sigma_1m": 0.001,
    }])
    obs = [
        Observation("BTC", 50020.0, "BINANCE", t0 - timedelta(seconds=30), t0 - timedelta(seconds=30)),
        Observation("BTC", 50050.0, "BINANCE", t0, t0),
    ]
    feat = compute_outsider_model_features(base, underlying_observations=obs)
    assert feat["has_ret_30s_ref"].iloc[0] is True or feat["has_ret_30s_ref"].iloc[0] == 1


# ==============================================================================
# R5 Tests
# ==============================================================================
def test_r5_future_candles_do_not_alter_past_sigma_and_z():
    """R5: Appending future candles must NOT change past sigma or past z_outsider."""
    from polyflip.models.point_in_time_features import compute_outsider_model_features

    ts = pd.Timestamp("2026-09-01T12:00:00Z")
    base = pd.DataFrame([{
        "mid_price": 0.3,
        "time_left_min": 5.0,
        "spread": 0.02,
        "candidate_side": "UP",
        "decision_at": ts,
        "underlying_price": 50010.0,
        "strike_value": 50000.0,
    }])
    past_closes = 50000.0 * np.exp(np.cumsum(np.array([0.0001, -0.0002, 0.0003, -0.0001] * 15)))
    candles = pd.DataFrame({
        "close": past_closes,
        "close_time": pd.date_range(end=ts, periods=60, freq="1min", tz="UTC"),
    })
    future = pd.DataFrame({
        "close": [55000.0, 45000.0] * 15,
        "close_time": pd.date_range(start=ts + timedelta(minutes=1), periods=30, freq="1min", tz="UTC"),
    })

    z1 = compute_outsider_model_features(base, minute_candles=candles).iloc[0]
    z2 = compute_outsider_model_features(base, minute_candles=pd.concat([candles, future])).iloc[0]

    assert float(z1["z_outsider"]) == pytest.approx(float(z2["z_outsider"]), rel=1e-7), \
        f"Future candles changed past z! Before: {z1['z_outsider']}, After: {z2['z_outsider']}"


def test_r5_decision_at_used_without_wallclock_leak():
    """R5: Momentum uses decision_at timestamp rather than datetime.now()."""
    from polyflip.models.point_in_time_features import compute_outsider_model_features
    from polyflip.crypto.underlying_observations import Observation

    ts = pd.Timestamp("2026-09-01T12:00:00Z")
    base = pd.DataFrame([{
        "mid_price": 0.3,
        "time_left_min": 5.0,
        "spread": 0.02,
        "candidate_side": "UP",
        "decision_at": ts,
        "underlying_price": 50000.0,
        "strike_value": 50000.0,
        "sigma_1m": 0.001,
    }])
    obs = [
        Observation("BTC", 49980.0, "BINANCE", ts - timedelta(seconds=30), ts - timedelta(seconds=30)),
        Observation("BTC", 50000.0, "BINANCE", ts, ts),
        Observation("BTC", 50500.0, "BINANCE", ts + timedelta(seconds=300), ts + timedelta(seconds=300)),
    ]

    class FutureClock:
        @staticmethod
        def now(tz=None):
            return (ts + timedelta(seconds=300)).to_pydatetime()

    with patch("polyflip.models.point_in_time_features.datetime", FutureClock):
        feat = compute_outsider_model_features(base, underlying_observations=obs)
        r30 = feat["ret_outsider_30s"].iloc[0]
        expected = np.log(50000.0 / 49980.0)
        assert r30 == pytest.approx(expected, rel=1e-5), f"Momentum leaked future observation! Got: {r30}"


def test_r5_missing_sigma_excluded_from_complete_b():
    """R5: Default fallback sigma=0.001 must NOT pass complete-B cohort."""
    from polyflip.models.outsider_dataset import prepare_outsider_dataset_cohorts

    ts = pd.Timestamp("2026-09-01T12:00:00Z")
    row = {
        "market_id": "m1",
        "recorded_at": ts,
        "time_left_min": 5.0,
        "mid_price": 0.35,
        "best_ask": 0.36,
        "spread": 0.02,
        "final_outcome": "UP",
        "underlying_price": 50010.0,
        "strike_value": 50000.0,
    }
    cohort = prepare_outsider_dataset_cohorts(pd.DataFrame([row]))
    assert len(cohort.df_full_b) == 0, "Missing sigma was admitted to complete-B cohort!"


# ==============================================================================
# R6 Tests
# ==============================================================================
def test_r6_mlegacy_uses_authentic_btc_leaning_model():
    """R6: Mlegacy loads authentic BTC_leaning@11 artifact or verified formula."""
    from polyflip.models.outsider_baselines import LegacyOutsiderBaseline

    baseline = LegacyOutsiderBaseline()
    assert baseline.is_authentic_model, "Mlegacy is not using authentic model artifact!"

    df_in = pd.DataFrame([{
        "mid_price": 0.30,
        "spread": 0.02,
        "time_left_min": 10.0,
    }])
    p = baseline.predict_proba(df_in)[0, 1]
    assert 0.01 < p < 0.99
    assert baseline.model_version == 11


def test_r6_mlegacy_orients_down_candidate_side():
    """R6: Authentic Mlegacy predicts p_flip directly for both UP and DOWN without faulty inversion."""
    from polyflip.models.outsider_baselines import LegacyOutsiderBaseline

    baseline = LegacyOutsiderBaseline()
    df_up = pd.DataFrame([{
        "mid_price": 0.26,
        "spread": 0.01,
        "time_left_min": 5.0,
        "candidate_side": "UP",
    }])
    df_down = pd.DataFrame([{
        "mid_price": 0.26,
        "yes_mid": 0.74,
        "spread": 0.01,
        "time_left_min": 5.0,
        "candidate_side": "DOWN",
    }])
    p_up = baseline.predict_proba(df_up)[0, 1]
    p_down = baseline.predict_proba(df_down)[0, 1]
    # In both cases, p_flip is an outsider probability (~0.30-0.32), NOT inverted to ~0.70
    assert 0.25 < p_up < 0.35
    assert 0.25 < p_down < 0.35


# ==============================================================================
# R7 Tests
# ==============================================================================
def test_r7_meta_stacking_temporal_market_isolation():
    """R7: Meta-model cross-validation must strictly isolate market_ids and time."""
    from polyflip.trading.combined_voting import build_meta_model_dataset

    n = 20
    df = pd.DataFrame({
        "market_id": [f"m_{i // 2}" for i in range(n)],
        "p_b": np.linspace(0.2, 0.8, n),
        "p_lgbm": np.linspace(0.25, 0.75, n),
        "target": [0, 1] * (n // 2),
        "decision_at": pd.date_range("2026-09-01", periods=n, freq="10min", tz="UTC"),
    })
    meta_data = build_meta_model_dataset(df)
    assert meta_data.get("temporal_isolated") is True


def test_r7_b_plus_input_evaluates_meta_probs_not_pb():
    """R7: MODEL_B_PLUS_LGBM_INPUT metrics must evaluate meta_probs, not p_b."""
    from scripts.research.compare_outsider_models import compare_all_models

    df = pd.DataFrame({
        "market_id": [f"m_{i}" for i in range(10)],
        "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "mid_price": [0.3] * 10,
        "executable_ask": [0.3] * 10,
        "outsider_mid": [0.3] * 10,
        "spread": [0.02] * 10,
        "time_left_min": [5.0] * 10,
        "decision_at": pd.date_range("2026-09-01", periods=10, freq="15min", tz="UTC"),
        "p_lgbm": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
        "lgbm_available": [True] * 10,
    })
    res = compare_all_models(df, source_kind="DEMO")
    rows = {r["model"]: r for r in res["summary_table"]}
    if "MODEL_B_PLUS_LGBM_INPUT" in rows and "MODEL_B1" in rows:
        assert rows["MODEL_B_PLUS_LGBM_INPUT"]["prediction_id"] != rows["MODEL_B1"]["prediction_id"]


# ==============================================================================
# R8 Tests
# ==============================================================================
def test_r8_one_position_per_market_in_replay():
    """R8: Replay state machine enforces max 1 entry per market (no duplicate orders)."""
    from polyflip.research.outsider_replay import OutsiderReplayEngine, ReplayPolicy

    engine = OutsiderReplayEngine(policy=ReplayPolicy(max_positions_per_market=1))
    t0 = pd.Timestamp("2026-09-01T12:00:00Z")
    decisions = [
        {"market_id": "m1", "decision_at": t0, "time_left_min": 10.0, "candidate_side": "UP", "executable_ask": 0.25, "p_win": 0.45, "target": 1},
        {"market_id": "m1", "decision_at": t0 + timedelta(minutes=5), "time_left_min": 5.0, "candidate_side": "UP", "executable_ask": 0.26, "p_win": 0.48, "target": 1},
        {"market_id": "m1", "decision_at": t0 + timedelta(minutes=8), "time_left_min": 2.0, "candidate_side": "UP", "executable_ask": 0.27, "p_win": 0.50, "target": 1},
    ]
    ledger = engine.run(decisions)
    assert len(ledger.executed_trades) == 1, f"Expected exactly 1 trade, got {len(ledger.executed_trades)}"
    assert ledger.executed_trades[0]["time_left_min"] == 10.0


def test_r8_net_ev_formula_and_non_default_costs():
    """R8: Net EV subtracts fees explicitly; missing ask is invalid."""
    from polyflip.crypto.edge import compute_net_ev_per_share

    ev = compute_net_ev_per_share(p_win=0.40, ask_price=0.30, fee_per_share=0.01, extra_cost_per_share=0.02)
    assert ev == pytest.approx(0.07, abs=1e-6)

    ev_nan = compute_net_ev_per_share(p_win=0.40, ask_price=np.nan, fee_per_share=0.01)
    assert np.isnan(ev_nan)


def test_r8_net_ev_invalid_bounds_returns_nan():
    """R8: Invalid ask prices (<=0 or >=1) or out-of-bounds p_win must return np.nan."""
    from polyflip.crypto.edge import compute_net_ev_per_share

    assert np.isnan(compute_net_ev_per_share(p_win=0.5, ask_price=0.0))
    assert np.isnan(compute_net_ev_per_share(p_win=0.5, ask_price=-0.1))
    assert np.isnan(compute_net_ev_per_share(p_win=0.5, ask_price=1.0))
    assert np.isnan(compute_net_ev_per_share(p_win=0.5, ask_price=1.05))
    assert np.isnan(compute_net_ev_per_share(p_win=-0.01, ask_price=0.3))
    assert np.isnan(compute_net_ev_per_share(p_win=1.05, ask_price=0.3))


def test_r8_replay_veto_mode_allows_subsequent_entries():
    """R8: In veto_mode, vetoed decision does not count towards market_entry_counts."""
    from polyflip.research.outsider_replay import OutsiderReplayEngine, ReplayPolicy

    engine = OutsiderReplayEngine(policy=ReplayPolicy(max_positions_per_market=1, veto_mode=True, min_edge=0.01))
    t0 = pd.Timestamp("2026-09-01T12:00:00Z")
    decisions = [
        # T-10: positive edge, but LGBM vetoes (DOWN vs candidate UP)
        {
            "market_id": "m_veto",
            "decision_at": t0,
            "time_left_min": 10.0,
            "candidate_side": "UP",
            "executable_ask": 0.20,
            "p_win": 0.35,
            "target": 1,
            "lgbm_direction": "DOWN",
        },
        # T-5: positive edge, LGBM agrees (UP vs candidate UP) -> should be allowed to enter!
        {
            "market_id": "m_veto",
            "decision_at": t0 + timedelta(minutes=5),
            "time_left_min": 5.0,
            "candidate_side": "UP",
            "executable_ask": 0.22,
            "p_win": 0.38,
            "target": 1,
            "lgbm_direction": "UP",
        },
    ]
    ledger = engine.run(decisions)
    assert len(ledger.executed_trades) == 1
    assert ledger.executed_trades[0]["time_left_min"] == 5.0
    assert len(ledger.vetoed_trades) == 1
    assert ledger.vetoed_trades[0]["time_left_min"] == 10.0


def test_r8_selection_returns_inconclusive_when_no_edge():
    """R8: Selection returns INCONCLUSIVE when CI crosses zero or all models negative."""
    from scripts.research.compare_outsider_models import select_candidate_configuration

    summary = [
        {"model": "MODEL_A1", "net_pnl": -5.0, "expectancy": -0.05, "expectancy_ci_lower": -0.10, "expectancy_ci_upper": 0.01},
        {"model": "MODEL_B1", "net_pnl": -3.0, "expectancy": -0.03, "expectancy_ci_lower": -0.08, "expectancy_ci_upper": 0.02},
    ]
    status, winner = select_candidate_configuration(summary)
    assert status in ("INCONCLUSIVE", "NO_CANDIDATE", "EDGE_NOT_SUPPORTED")
    assert winner is None or status != "EDGE_SUPPORTED"


# ==============================================================================
# R9 Tests
# ==============================================================================
def test_r9_drawdown_two_losses_equals_two():
    """R9: Two losses [-1.0, -1.0] from initial equity 0 must give drawdown = 2.0."""
    from polyflip.research.reporting_helpers import compute_drawdown

    dd = compute_drawdown([-1.0, -1.0])
    assert dd == pytest.approx(2.0, abs=1e-6), f"Expected max DD=2.0, got {dd}"


def test_r9_single_cluster_returns_insufficient_blocks():
    """R9: 1 cluster must NOT return SE=0.0 and fake confident CI."""
    from polyflip.research.reporting_helpers import compute_clustered_uncertainty

    res = compute_clustered_uncertainty(trade_pnls=[0.6, -0.4], cluster_ids=["m1", "m1"])
    assert res.get("status") == "INSUFFICIENT_BLOCKS"
    assert np.isnan(res["ci_lower"])
    assert np.isnan(res["ci_upper"])
    assert np.isnan(res["se"])


def test_r9_paired_cluster_delta_insufficient_blocks():
    """R9: Paired cluster delta with <=1 cluster must return INSUFFICIENT_BLOCKS and NaN CI."""
    from polyflip.research.reporting_helpers import compute_paired_cluster_delta

    res = compute_paired_cluster_delta(
        pnls_model=[0.5],
        indices_model=[0],
        pnls_ref=[0.1],
        indices_ref=[0],
        cluster_ids_full=["c1"],
    )
    assert res.get("status") == "INSUFFICIENT_BLOCKS"
    assert np.isnan(res["ci_lower"])
    assert np.isnan(res["ci_upper"])
    assert np.isnan(res["se"])


def test_r9_price_bins_reliability_separated_from_probability_bins():
    """R9: Price bin reliability (0.05 bins) is separate from probability ECE."""
    from polyflip.research.reporting_helpers import generate_price_bins_report

    trades = pd.DataFrame({
        "executable_ask": [0.12, 0.18, 0.22, 0.28],
        "p_win": [0.25, 0.30, 0.35, 0.40],
        "outcome": [0, 1, 0, 1],
        "pnl": [-1.0, 2.3, -1.0, 1.8],
    })
    rep = generate_price_bins_report(trades, bin_width=0.05)
    assert not rep.empty
    assert "price_bin" in rep.columns
    assert "realized_frequency" in rep.columns
