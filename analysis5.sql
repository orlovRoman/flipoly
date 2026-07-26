SELECT asset, version, trained_at, accuracy, ece, backtest_pnl, backtest_trades, backtest_wr 
FROM model_registry 
WHERE is_active = true AND asset IN ('BTC', 'ETH', 'SOL', 'XRP');
