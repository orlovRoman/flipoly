SELECT asset, accuracy, ece 
FROM model_registry 
WHERE is_active = true AND asset IN ('BTC', 'ETH', 'SOL', 'XRP');
