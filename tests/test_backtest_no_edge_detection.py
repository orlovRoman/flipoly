import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    n = 2000
    np.random.seed(1)
    df = pd.DataFrame({
        "ret_1": np.random.randn(n) * 0.002,
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min"),
    })
    from polyflip.crypto.trainer import CRYPTO_FEATURES
    for f in CRYPTO_FEATURES:
        if f not in df.columns:
            df[f] = np.random.randn(n)
    return df

def test_no_edge_model_is_marked_not_profitable(sample_df):
    """
    Модель с AUC < 0.53 (нет edge) должна вернуть is_profitable() = False.
    """
    from polyflip.crypto.backtester import run_backtest
    # Полностью рандомные данные → AUC ~0.5
    result = run_backtest(sample_df, "ETHUSDT", min_edge=0.05, commission=0.001)
    
    if result.train_auc < 0.53:
        assert not result.is_profitable(), (
            f"AUC={result.train_auc:.3f} — модель без edge должна быть is_profitable()=False"
        )

def test_sharpe_negative_when_net_return_negative(sample_df):
    """Отрицательный доход → Sharpe должен быть отрицательным."""
    from polyflip.crypto.backtester import run_backtest
    result = run_backtest(sample_df, "ETHUSDT", commission=0.002)  # высокая комиссия
    
    if result.total_return_net < 0:
        assert result.sharpe_ratio < 0, (
            f"total_return={result.total_return_net:.2%} отрицательный, "
            f"но Sharpe={result.sharpe_ratio:.2f} положительный — ошибка расчёта"
        )
