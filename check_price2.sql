SELECT id, asset, status, executed_price, amount_usdc, entry_filled_shares 
FROM trade_history 
WHERE status = 'SUCCESS' 
ORDER BY created_at DESC 
LIMIT 5;
