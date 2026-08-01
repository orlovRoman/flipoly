cd /home/orlovrp/flipoly
docker compose exec -T db sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT id, state, filled_shares, filled_cost_usdc
FROM execution_requests
WHERE id = 'ca9d260c-47fb-4e21-bb9f-aa33cbe818d7';

SELECT request_id, released_at
FROM exposure_reservations
WHERE request_id = 'ca9d260c-47fb-4e21-bb9f-aa33cbe818d7';
SQL
