\x on
SELECT * FROM trade_history 
WHERE asset = 'SOL' 
  AND status = 'SUCCESS'
  AND (executed_price = 0.55 OR ABS(pnl - 0.8181818) < 0.05 OR created_at >= '2026-07-29 00:00:00+00')
ORDER BY created_at DESC;
