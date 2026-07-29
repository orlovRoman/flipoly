\x on
SELECT * FROM execution_requests WHERE trade_history_id = 31965;

SELECT * FROM execution_attempts WHERE request_id IN (SELECT id FROM execution_requests WHERE trade_history_id = 31965);

SELECT * FROM execution_fills WHERE attempt_id IN (
    SELECT ea.id FROM execution_attempts ea 
    JOIN execution_requests er ON ea.request_id = er.id 
    WHERE er.trade_history_id = 31965
);
