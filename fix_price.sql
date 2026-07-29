UPDATE trade_history 
SET executed_price = amount_usdc / entry_filled_shares 
WHERE executed_price = 0 
  AND entry_filled_shares > 0 
  AND status IN ('SUCCESS', 'EXECUTED');
