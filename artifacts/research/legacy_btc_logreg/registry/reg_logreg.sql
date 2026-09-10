SET statement_timeout TO '120s';
SELECT asset, COUNT(*) FROM model_registry WHERE model_type = 'logreg' GROUP BY 1 ORDER BY 1;
\copy (SELECT id, asset, version, model_type, trained_at, training_window_start, training_window_end, activated_at, is_active, train_samples, backtest_pnl, backtest_trades, backtest_wr, brier_score, accuracy, decision_threshold, decision_threshold_down, features FROM model_registry WHERE model_type = 'logreg' ORDER BY asset, version, id) TO '/tmp/reg_logreg.csv' WITH (FORMAT CSV, HEADER)
