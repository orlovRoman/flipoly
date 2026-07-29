\x on
SELECT * FROM trade_history 
WHERE asset = 'SOL' 
  AND created_at >= '2026-07-28 14:00:00+00' 
  AND created_at <= '2026-07-28 15:00:00+00'
ORDER BY created_at DESC 
LIMIT 5;
