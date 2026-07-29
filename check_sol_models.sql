SELECT id, name, asset, version, is_active, created_at 
FROM model_registry 
WHERE name LIKE '%SOL%' OR asset LIKE '%SOL%' OR asset LIKE '%sol%'
ORDER BY created_at DESC;

SELECT asset, model_version, active_features, count(*), sum(coalesce(pnl, 0)) as total_pnl 
FROM trade_history 
WHERE asset LIKE '%SOL%' OR asset LIKE '%sol%'
GROUP BY asset, model_version, active_features;
