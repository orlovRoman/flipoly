\x on
SELECT * FROM trade_history 
WHERE asset = 'ETH' 
  AND created_at >= '2026-07-29 07:45:00+00'
  AND created_at <= '2026-07-29 08:05:00+00';
