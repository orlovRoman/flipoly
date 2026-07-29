SELECT id, asset, outcome_bought, amount_usdc, executed_price, status, pnl, created_at, active_features, model_version
FROM trade_history 
WHERE asset = 'SOL' 
  AND created_at >= '2026-07-29 13:00:00+00'
ORDER BY created_at DESC;
