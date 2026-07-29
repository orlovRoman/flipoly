SELECT 
    asset,
    CASE 
        WHEN active_features LIKE '%outsider%' THEN 'Аутсайдер'
        WHEN active_features LIKE '%favorite%' OR active_features LIKE '%ml_trend%' THEN 'Фаворит'
        ELSE 'Другое'
    END as strategy,
    mode,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as win_rate,
    SUM(COALESCE(pnl, 0)) as total_pnl,
    MIN(created_at) as earliest_trade,
    MAX(created_at) as latest_trade
FROM trade_history
WHERE asset = 'SOL' 
  AND created_at >= NOW() - INTERVAL '24 HOURS'
  AND status = 'SUCCESS'
GROUP BY asset, strategy, mode;
