from polyflip.crypto.backtester import BacktestResult

# profitable: sharpe > 0.5 и win_rate > 0.52
r1 = BacktestResult(
    symbol="X",
    n_candles_total=1000,
    n_candles_test=300,
    n_trades=150,
    win_rate=0.56,
    total_return=0.02,
    total_return_net=0.015,
    sharpe_ratio=0.8,
    max_drawdown=-0.03,
    edge_rate=0.12,
    epsilon=0.001,
    train_auc=0.58,
)
assert r1.is_profitable(), "Должен быть profitable (Sharpe=0.8, WinRate=56%)"

# not profitable: sharpe < 0.5 и win_rate < 0.52
r2 = BacktestResult(
    symbol="X",
    n_candles_total=1000,
    n_candles_test=300,
    n_trades=10,
    win_rate=0.48,
    total_return=0.001,
    total_return_net=-0.005,
    sharpe_ratio=0.1,
    max_drawdown=-0.15,
    edge_rate=0.01,
    epsilon=0.001,
    train_auc=0.51,
)
assert not r2.is_profitable(), "Не должен быть profitable (Sharpe=0.1, WinRate=48%)"

# edge case: нет сделок
r3 = BacktestResult(
    symbol="ETH",
    n_candles_total=500,
    n_candles_test=150,
    n_trades=0,
    win_rate=0.0,
    total_return=0.0,
    total_return_net=0.0,
    sharpe_ratio=0.0,
    max_drawdown=0.0,
    edge_rate=0.0,
    epsilon=0.002,
    train_auc=0.50,
)
assert not r3.is_profitable(), "Не должен быть profitable (0 сделок)"

print("BacktestResult thresholds OK")
print(r1.summary())
print(r2.summary())
print(r3.summary())
