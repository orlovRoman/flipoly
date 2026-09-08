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
        "time": dates,
        "market": markets,
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
    
    # Change target for the second half of the data
    df_modified = df.copy()
    df_modified.loc[50:, "target"] = 1 - df_modified.loc[50:, "target"]
    
    # Recalculate
    res2 = evaluate_lgbm_outsider_interaction(df_modified, min_edge=0.01)
    probs2 = res2["b_plus_lgbm_input"]["meta_probs"]
    
    # Check if first 20 predictions changed
    # They should NOT change if there's no future data leakage
    # In the bugged version, GroupKFold leaks future changes into past predictions
    assert np.allclose(probs1[:20], probs2[:20], equal_nan=True), "Future data leaked into past predictions!"
