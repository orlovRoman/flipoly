DELETE FROM execution_attempts WHERE request_id IN (
    SELECT id FROM execution_requests WHERE trade_history_id IN (
        SELECT id FROM trade_history WHERE market_id = 'test_market' OR market_id LIKE 'test_%' OR asset = 'USDC'
    )
);
DELETE FROM execution_requests WHERE trade_history_id IN (
    SELECT id FROM trade_history WHERE market_id = 'test_market' OR market_id LIKE 'test_%' OR asset = 'USDC'
);
DELETE FROM trade_history WHERE market_id = 'test_market' OR market_id LIKE 'test_%' OR asset = 'USDC';
