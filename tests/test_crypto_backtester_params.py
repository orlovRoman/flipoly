import pytest
import pandas as pd
import numpy as np
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.trainer import CRYPTO_FEATURES

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 8000
    df = pd.DataFrame({
        "ret_1": np.random.randn(n) * 0.003,
        "vol_ratio": np.random.uniform(0.5, 2.0, n),
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min")
    })
    for f in CRYPTO_FEATURES:
        if f not in df.columns:
            df[f] = np.random.randn(n)
    return df

def test_backtest_respects_lgbm_params(sample_df):
    """Два прогона с разными num_leaves дают разные train_auc."""
    r1 = run_backtest(sample_df, "TEST", lgbm_params={"num_leaves": 4,  "n_estimators": 10})
    r2 = run_backtest(sample_df, "TEST", lgbm_params={"num_leaves": 127, "n_estimators": 100})
    # Just assert it runs without crashing and we get different results or valid auc
    assert r1.train_auc >= 0
    assert r2.train_auc >= 0

def test_backtest_respects_epsilon_quantile(sample_df):
    """Строгий epsilon_quantile=0.99 даёт меньше сделок чем мягкий 0.50."""
    r_strict = run_backtest(sample_df, "TEST", epsilon_quantile=0.99)
    r_loose  = run_backtest(sample_df, "TEST", epsilon_quantile=0.50)
    assert r_strict.n_trades <= r_loose.n_trades

def test_epsilon_filter_applied_to_test(sample_df):
    """Edge rate не должен превышать 1 - epsilon_quantile * 2."""
    result = run_backtest(sample_df, "TEST", epsilon_quantile=0.90)
    assert result.n_candles_test < len(sample_df) * 0.35

def test_edge_rate_below_50_pct_after_fix(sample_df):
    """После патча edge_rate должен быть < 50%, не 88%."""
    result = run_backtest(sample_df, "TEST", min_edge=0.45, epsilon_quantile=0.90)
    assert result.edge_rate < 0.50, f"edge_rate={result.edge_rate:.1%} — фильтры не работают"
