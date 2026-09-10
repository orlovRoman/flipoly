SET statement_timeout TO '60s';
SELECT id, asset, version, trained_at, training_window_start, training_window_end,
       activated_at, is_active, train_samples, backtest_pnl, backtest_trades,
       backtest_wr, brier_score, ece, accuracy, decision_threshold,
       decision_threshold_down
  FROM model_registry WHERE asset = 'BTC_leaning' ORDER BY version;
SELECT id, asset, version, trained_at, activated_at, is_active, backtest_pnl,
       backtest_trades, ece FROM model_registry
  WHERE asset IN ('ETH_leaning','SOL_leaning','XRP_leaning','DOGE_leaning')
    AND trained_at BETWEEN timestamptz '2026-08-10 00:00+00' AND timestamptz '2026-08-22 00:00+00'
  ORDER BY asset, trained_at;
