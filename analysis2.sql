SELECT mode, asset, COUNT(*) as cnt, COUNT(pnl) as resolved_cnt
FROM trade_history
WHERE created_at >= NOW() - INTERVAL '48 hours'
GROUP BY mode, asset;
