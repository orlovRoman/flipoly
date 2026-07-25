SELECT 
  error_msg, 
  COUNT(*) 
FROM trade_history 
WHERE status = 'SKIPPED'
  AND (error_msg LIKE '%favorite%' 
    OR error_msg LIKE '%time%' 
    OR error_msg LIKE '%edge%' 
    OR error_msg LIKE '%p_flip%' 
    OR error_msg LIKE '%threshold%' 
    OR error_msg LIKE '%window%')
GROUP BY error_msg 
ORDER BY COUNT(*) DESC 
LIMIT 25;
