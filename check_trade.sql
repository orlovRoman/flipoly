SELECT * FROM trade_history 
WHERE asset = 'ETH' 
  AND created_at >= '2026-07-29 07:00:00+00'
ORDER BY created_at DESC 
LIMIT 5;
