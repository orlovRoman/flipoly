-- Удаляем в правильном порядке по FK цепочке

-- 1. execution_fills -> execution_attempts -> execution_requests -> trade_history
-- Находим все request_id связанные со старыми сделками
DELETE FROM execution_fills
WHERE attempt_id IN (
    SELECT ea.id FROM execution_attempts ea
    JOIN execution_requests er ON ea.request_id = er.id
    WHERE er.trade_history_id IN (
        SELECT id FROM trade_history WHERE created_at < '2026-07-28 00:00:00+00'
    )
);

-- 2. Удаляем execution_attempts
DELETE FROM execution_attempts
WHERE request_id IN (
    SELECT id FROM execution_requests
    WHERE trade_history_id IN (
        SELECT id FROM trade_history WHERE created_at < '2026-07-28 00:00:00+00'
    )
);

-- 3. Удаляем execution_approvals (если есть)
DELETE FROM execution_approvals
WHERE request_id IN (
    SELECT id FROM execution_requests
    WHERE trade_history_id IN (
        SELECT id FROM trade_history WHERE created_at < '2026-07-28 00:00:00+00'
    )
);

-- 4. Удаляем execution_events (если есть)
DELETE FROM execution_events
WHERE request_id IN (
    SELECT id FROM execution_requests
    WHERE trade_history_id IN (
        SELECT id FROM trade_history WHERE created_at < '2026-07-28 00:00:00+00'
    )
);

-- 5. Удаляем execution_requests
DELETE FROM execution_requests
WHERE trade_history_id IN (
    SELECT id FROM trade_history WHERE created_at < '2026-07-28 00:00:00+00'
);

-- 6. Наконец удаляем сами сделки
DELETE FROM trade_history 
WHERE created_at < '2026-07-28 00:00:00+00';

-- Финальная проверка
SELECT COUNT(*) as total_remaining FROM trade_history;
SELECT MIN(created_at) as oldest_record FROM trade_history;
