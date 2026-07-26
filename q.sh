docker compose exec -T db psql -U polyflip -d polyflip << 'SQL'
SELECT id, pnl, amount_usdc, realized_pnl_usdc, status, created_at, entry_cost_usdc FROM trade_history WHERE status='SUCCESS' ORDER BY created_at DESC LIMIT 10;
SQL
