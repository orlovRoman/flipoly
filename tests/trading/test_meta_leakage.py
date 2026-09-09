import numpy as np
import pandas as pd
import pytest
from polyflip.trading.combined_voting import evaluate_lgbm_outsider_interaction


def test_meta_model_time_leakage():
    # Create a dummy dataset over several days
    n_rows = 100
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="1h", tz="UTC")
    markets = [f"market_{i//5}" for i in range(n_rows)]
    
    df = pd.DataFrame({
        "decision_at": dates,
        "market_id": markets,
        "is_out_of_sample": True,
        "training_cutoff_at": dates - pd.Timedelta(hours=1),
        "p_b_win": np.random.uniform(0.1, 0.9, n_rows),
        "lgbm_oof_prob": np.random.uniform(0.1, 0.9, n_rows),
        "target": np.random.randint(0, 2, n_rows),
        "executable_ask": np.full(n_rows, 0.4),
        "candidate_side": ["UP"] * n_rows,
        "lgbm_direction": ["UP"] * n_rows,
    })
    
    # Calculate original meta_probs
    res1 = evaluate_lgbm_outsider_interaction(df, min_edge=0.01)
    probs1 = res1["b_plus_lgbm_input"]["meta_probs"]
    
    # Ensure real non-NaN predictions are actually formed
    assert np.isfinite(probs1[20:]).sum() > 0, "Meta predictions should be generated for walk-forward folds"
    
    # Change target for the second half of the data
    df_modified = df.copy()
    df_modified.loc[50:, "target"] = 1 - df_modified.loc[50:, "target"]
    
    # Recalculate
    res2 = evaluate_lgbm_outsider_interaction(df_modified, min_edge=0.01)
    probs2 = res2["b_plus_lgbm_input"]["meta_probs"]
    
    # Check if first 20 predictions changed
    # They should NOT change if there's no future data leakage
    assert np.allclose(probs1[:20], probs2[:20], equal_nan=True), "Future data leaked into past predictions!"


def test_meta_model_rejects_in_sample_or_future_predictions():
    """Item 6: Contract must reject in-sample predictions or future training cutoff."""
    n_rows = 50
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="1h", tz="UTC")
    markets = [f"market_{i//5}" for i in range(n_rows)]

    # 1. In-sample predictions rejected
    df_insample = pd.DataFrame({
        "decision_at": dates,
        "market_id": markets,
        "is_out_of_sample": False,
        "training_cutoff_at": dates - pd.Timedelta(hours=1),
        "p_b_win": np.random.uniform(0.1, 0.9, n_rows),
        "lgbm_oof_prob": np.random.uniform(0.1, 0.9, n_rows),
        "target": np.random.randint(0, 2, n_rows),
        "executable_ask": np.full(n_rows, 0.4),
        "candidate_side": ["UP"] * n_rows,
        "lgbm_direction": ["UP"] * n_rows,
    })
    res_in = evaluate_lgbm_outsider_interaction(df_insample, min_edge=0.01)
    assert res_in["b_plus_lgbm_input"]["meta_status"] == "IN_SAMPLE_PREDICTIONS_REJECTED"
    assert res_in["b_plus_lgbm_input"]["n_trades"] == 0

    # 2. Future predictions rejected
    df_future = df_insample.copy()
    df_future["is_out_of_sample"] = True
    df_future["training_cutoff_at"] = dates + pd.Timedelta(hours=2)
    res_fut = evaluate_lgbm_outsider_interaction(df_future, min_edge=0.01)
    assert res_fut["b_plus_lgbm_input"]["meta_status"] == "FUTURE_PREDICTIONS_REJECTED"
    assert res_fut["b_plus_lgbm_input"]["n_trades"] == 0


def test_meta_model_label_availability_prevents_training():
    """Item 4: If labels are not available yet (e.g. 2099-01-01), meta-model cannot train."""
    n_rows = 50
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="1h", tz="UTC")
    markets = [f"market_{i//5}" for i in range(n_rows)]

    df = pd.DataFrame({
        "decision_at": dates,
        "market_id": markets,
        "is_out_of_sample": True,
        "training_cutoff_at": dates - pd.Timedelta(hours=1),
        "label_available_at": pd.to_datetime(["2099-01-01"] * n_rows, utc=True),
        "p_b_win": np.random.uniform(0.1, 0.9, n_rows),
        "lgbm_oof_prob": np.random.uniform(0.1, 0.9, n_rows),
        "target": np.random.randint(0, 2, n_rows),
        "executable_ask": np.full(n_rows, 0.4),
        "candidate_side": ["UP"] * n_rows,
        "lgbm_direction": ["UP"] * n_rows,
    })
    res = evaluate_lgbm_outsider_interaction(df, min_edge=0.01)
    assert res["b_plus_lgbm_input"]["meta_status"] == "NO_LABELS_AVAILABLE"
    assert res["b_plus_lgbm_input"]["n_trades"] == 0
    assert np.isnan(res["b_plus_lgbm_input"]["meta_probs"]).all()


def test_meta_model_rejects_missing_temporal_fields():
    """Item 5: No fake dates or silent fallbacks if decision_at or market_id missing."""
    n_rows = 50
    df_no_time = pd.DataFrame({
        "market_id": [f"market_{i//5}" for i in range(n_rows)],
        "is_out_of_sample": True,
        "training_cutoff_at": pd.date_range("2026-01-01", periods=n_rows, freq="1h", tz="UTC"),
        "p_b_win": np.random.uniform(0.1, 0.9, n_rows),
        "lgbm_oof_prob": np.random.uniform(0.1, 0.9, n_rows),
        "target": np.random.randint(0, 2, n_rows),
        "executable_ask": np.full(n_rows, 0.4),
        "candidate_side": ["UP"] * n_rows,
        "lgbm_direction": ["UP"] * n_rows,
    })
    res = evaluate_lgbm_outsider_interaction(df_no_time, min_edge=0.01)
    assert res["b_plus_lgbm_input"]["meta_status"] == "MISSING_TEMPORAL_FIELDS"
    assert res["b_plus_lgbm_input"]["n_trades"] == 0
