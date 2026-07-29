SELECT id, asset, model_version, active_features, status, pnl, created_at 
FROM trade_history 
WHERE asset LIKE '%SOL%' AND created_at >= '2026-07-28 00:00:00+00'
ORDER BY created_at DESC;
