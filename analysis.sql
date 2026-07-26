-- Модели
SELECT asset, version, trained_at, accuracy, ece, features
FROM model_registry 
WHERE is_active = true 
ORDER BY asset;

-- Сделки (Аутсайдер)
SELECT asset, 
       COUNT(pnl) as cnt, 
       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(pnl) as win_rate,
       SUM(pnl) as total_pnl,
       AVG(edge) as avg_edge,
       AVG(predicted_flip_prob) as avg_p_flip,
       AVG(executed_price) as avg_buy_price,
       AVG(amount_usdc) as avg_bet_size
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND mode IN ('ml_trend', 'combined')
  AND pnl IS NOT NULL
GROUP BY asset
ORDER BY total_pnl DESC;

-- Детали по моделям и исходам (Выигравшие сделки Аутсайдер)
SELECT asset, AVG(edge) as win_avg_edge, AVG(executed_price) as win_avg_buy_price
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND mode IN ('ml_trend', 'combined')
  AND pnl > 0
GROUP BY asset;

-- Детали по моделям и исходам (Проигравшие сделки Аутсайдер)
SELECT asset, AVG(edge) as lose_avg_edge, AVG(executed_price) as lose_avg_buy_price
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND mode IN ('ml_trend', 'combined')
  AND pnl <= 0
GROUP BY asset;
