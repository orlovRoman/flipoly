\x on
SELECT id, market_id, asset, outcome_bought, amount_usdc, executed_price, predicted_flip_prob, status, mode, pnl, edge, created_at, closed_at, position_status, entry_filled_shares, entry_cost_usdc, remaining_shares, realized_pnl_usdc
FROM trade_history 
WHERE asset = 'ETH' 
  AND created_at >= '2026-07-29 07:45:00+00'
  AND created_at <= '2026-07-29 08:05:00+00';
