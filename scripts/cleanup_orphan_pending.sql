-- Шаг 1: просмотр
SELECT market_id, count(*) AS cnt, min(created_at), max(created_at) 
FROM trade_history 
WHERE status = 'PENDING' 
AND NOT EXISTS (
    SELECT 1 FROM execution_requests r WHERE r.trade_history_id = trade_history.id
)
GROUP BY market_id 
ORDER BY cnt DESC;

-- Шаг 2: применение (только после проверки шага 1)
UPDATE trade_history 
SET status = 'SKIPPED', position_status = 'ENTRY_FAILED', error_msg = 'Recovered orphan PENDING: execution request was not created'
WHERE status = 'PENDING' 
AND NOT EXISTS (
    SELECT 1 FROM execution_requests r WHERE r.trade_history_id = trade_history.id
);

-- Шаг 3: верификация
SELECT count(*) 
FROM trade_history 
WHERE status = 'PENDING' 
AND NOT EXISTS (
    SELECT 1 FROM execution_requests r WHERE r.trade_history_id = trade_history.id
);
-- Ожидается: 0
