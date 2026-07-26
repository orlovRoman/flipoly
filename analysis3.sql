SELECT 
    asset,
    COUNT(pnl) as trades, 
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(pnl) * 100 as win_rate,
    SUM(pnl) as total_pnl,
    AVG(edge) as avg_edge,
    AVG(predicted_flip_prob) as avg_p_flip,
    AVG(executed_price) as avg_buy_price
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND pnl IS NOT NULL
  AND executed_price < 0.5
GROUP BY asset
ORDER BY total_pnl DESC;

SELECT asset, AVG(edge) as avg_win_edge, AVG(executed_price) as avg_win_buy_price
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND pnl > 0
  AND executed_price < 0.5
GROUP BY asset;

SELECT asset, AVG(edge) as avg_lose_edge, AVG(executed_price) as avg_lose_buy_price
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND pnl <= 0
  AND executed_price < 0.5
GROUP BY asset;
