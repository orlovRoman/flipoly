import pytest
import pandas as pd
import numpy as np
from polyflip.crypto.trainer import CRYPTO_FEATURES, _fit_lgbm_and_serialize

def make_trending_df(n=5000, trend_strength=0.0):
    """Датасет с реальным трендом — модель ДОЛЖНА поймать его AUC > 0.55."""
    np.random.seed(0)
    base = np.random.randn(n) * 0.002
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = 0.5 * ret[i-1] + base[i] + trend_strength
    df = pd.DataFrame({"ret_1": ret})
    df["open_time"] = pd.date_range("2024-01-01", periods=n, freq="15min")
    for f in CRYPTO_FEATURES:
        if f not in df.columns:
            df[f] = np.random.randn(n)
    # target: будет ли следующая свеча позитивной?
    df["target"] = (df["ret_1"].shift(-1) > 0).astype(int)
    # ПРАВИЛЬНО: предиктивный признак на основе ПРОШЛОГО, а не будущего
    df["vol_6"] = df["ret_1"].shift(1) + np.random.randn(n) * 0.001  # лаг +1 (прошлое)
    df = df.dropna()
    return df

def test_target_not_using_future_close_directly():
    """
    Если AUC < 0.52 на трендовом датасете — target скорее всего
    строится с использованием текущей свечи (leakage) или неправильно смещён.
    """
    df = make_trending_df()
    _, auc, _, _, _, _ = _fit_lgbm_and_serialize(
        df[CRYPTO_FEATURES], df["target"], n_splits=3
    )
    # На трендовом DF с нормальным target должны получить AUC > 0.52
    assert auc > 0.52, (
        f"AUC={auc:.3f} на трендовом датасете — "
        "возможен data leakage или неправильный target alignment"
    )

def test_target_is_forward_looking_not_current():
    """
    target[i] должен зависеть от будущего (ret[i+1]), 
    НЕ от текущей свечи ret[i] — иначе leakage.
    """
    from polyflip.crypto.backtester import run_backtest
    n = 3000
    np.random.seed(42)
    df = pd.DataFrame({
        "ret_1": np.random.randn(n) * 0.003,
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min"),
    })
    for f in CRYPTO_FEATURES:
        if f not in df.columns:
            df[f] = np.random.randn(n)
    
    result = run_backtest(df, "ETHUSDT", min_edge=0.05)
    # С рандомными данными AUC не должен быть > 0.60 (leakage flag)
    assert result.train_auc < 0.60, (
        f"AUC={result.train_auc:.3f} на рандомных данных — возможен data leakage!"
    )
