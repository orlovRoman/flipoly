SELECT id, symbol, asset, version, regime, is_active, created_at, ece, val_auc, baseline_auc 
FROM model_registry 
WHERE symbol LIKE '%SOL%' OR asset LIKE '%SOL%'
ORDER BY created_at DESC;
