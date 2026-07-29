SELECT id, asset, status, executed_price, amount_usdc, 
       position_status, entry_filled_shares, entry_cost_usdc, created_at
FROM trade_history
ORDER BY created_at DESC
LIMIT 10;
