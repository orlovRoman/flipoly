SELECT asset, accuracy, ece 
FROM model_registry 
WHERE is_active = true 
ORDER BY asset;
