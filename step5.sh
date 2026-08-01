cd /home/orlovrp/flipoly
docker compose exec -T db sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT id, state, filled_shares, filled_cost_usdc
FROM execution_requests
WHERE id = '1b62298f-ce15-4d70-8165-6f3858bd2fd4';

SELECT id, status, position_status, remaining_shares, error_msg
FROM trade_history
WHERE id = 34353;

SELECT request_id, released_at
FROM exposure_reservations
WHERE request_id = '1b62298f-ce15-4d70-8165-6f3858bd2fd4';
SQL
