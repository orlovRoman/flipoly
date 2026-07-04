import numpy as np
import pandas as pd
from polyflip.crypto.backtester import run_backtest, BacktestResult

np.random.seed(42)
n = 6000  # нужно много строк: epsilon-фильтр (90-й перцентиль) оставляет ~10%
df = pd.DataFrame({
    "ret_1":           np.random.randn(n) * 0.003,
    "ret_3":           np.random.randn(n) * 0.005,
    "ret_6":           np.random.randn(n) * 0.007,
    "ret_12":          np.random.randn(n) * 0.009,
    "ret_24":          np.random.randn(n) * 0.012,
    "vol_6":           np.abs(np.random.randn(n)) * 0.002 + 0.001,
    "vol_24":          np.abs(np.random.randn(n)) * 0.003 + 0.001,
    "vol_48":          np.abs(np.random.randn(n)) * 0.003 + 0.001,
    "vol_ratio":       np.random.uniform(0.5, 2.0, n),
    "rsi_14":          np.random.uniform(30, 70, n),
    "ema_ratio_9_21":  np.random.uniform(0.99, 1.01, n),
    "bb_width":        np.random.uniform(0.01, 0.05, n),
    "bb_position":     np.random.uniform(0, 1, n),
    "taker_buy_ratio": np.random.uniform(0.4, 0.6, n),
    "hour_utc":        np.random.randint(0, 24, n).astype(float),
    "consec_up":       np.random.randint(0, 5, n).astype(float),
    "consec_down":     np.random.randint(0, 5, n).astype(float),
})

result = run_backtest(df, "BTCUSDT")

assert isinstance(result, BacktestResult),       "Результат должен быть BacktestResult"
assert result.n_candles_total == 6000,           f"n_candles_total={result.n_candles_total}"
assert 0.0 <= result.win_rate <= 1.0,            f"win_rate вне диапазона: {result.win_rate}"
assert result.epsilon > 0,                       f"epsilon <= 0: {result.epsilon}"
assert result.n_candles_test > 0,               f"нет тестовых свечей"

print(result.summary())
print("Backtester OK")
print(f"  n_trades={result.n_trades}, edge_rate={result.edge_rate:.1%}")
print(f"  train_auc={result.train_auc:.3f}, max_dd={result.max_drawdown:.4f}")
