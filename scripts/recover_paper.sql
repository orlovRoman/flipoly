WITH reset AS (
    UPDATE execution_requests r
    SET 
        state = 'READY',
        claimed_by = NULL,
        claimed_at = NULL,
        lease_expires_at = NULL,
        expires_at = now() + interval '5 minutes',
        updated_at = now(),
        error_reason = NULL
    WHERE r.requested_mode = 'PAPER'
      AND r.state = 'RECONCILING'
      AND NOT EXISTS (
          SELECT 1 FROM execution_attempts a
          JOIN execution_fills f ON f.attempt_id = a.id
          WHERE a.request_id = r.id
      )
    RETURNING r.id, r.trade_history_id
)
INSERT INTO execution_events (
    level, event_type, message, source, request_id, trade_history_id
)
SELECT 
    'WARNING', 
    'PAPER_REQUEST_REQUEUED', 
    'Requeued after fake settlement-state regression', 
    'recovery', 
    id, 
    trade_history_id
FROM reset;
