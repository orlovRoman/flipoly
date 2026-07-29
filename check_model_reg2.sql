SELECT id, asset, version, is_active, trained_at, accuracy, baseline, ece 
FROM model_registry 
WHERE asset LIKE '%SOL%' OR asset LIKE '%sol%'
ORDER BY trained_at DESC;
