import subprocess
import sys

verify_script = """cd /home/orlovrp/flipoly

echo "=== CHECKING CONTAINERS ==="
sudo docker compose ps api scheduler execution_worker_paper

sudo docker compose \\
  --env-file /home/orlovrp/.flipoly-live-v2.env \\
  -f docker-compose.live-v2.yml \\
  --profile live-v2 \\
  ps

echo "=== CHECKING LOGS ==="
sudo docker compose logs --no-color --since=3m api \\
  | grep -E "Traceback|ERROR|500 Internal" || true

sudo docker compose \\
  --env-file /home/orlovrp/.flipoly-live-v2.env \\
  -f docker-compose.live-v2.yml \\
  --profile live-v2 \\
  logs --no-color --since=3m \\
  execution_worker_live \\
  live_mirror_worker \\
  release_gate_worker \\
  | grep -E "Traceback|ERROR|exception|failed" || true

echo "=== CHECKING HEARTBEAT ==="
sudo docker compose exec -T db sh -lc \\
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT
    worker_id,
    execution_mode,
    gateway_ready,
    now() - heartbeat_at AS heartbeat_age,
    last_error_message
FROM execution_worker_status
WHERE execution_mode = 'LIVE'
ORDER BY heartbeat_at DESC
LIMIT 3;
SQL

echo "=== CHECKING CLEANUP ==="
sudo docker compose exec -T db sh -lc \\
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT state, count(*)
FROM execution_requests
WHERE requested_mode = 'LIVE'
GROUP BY state
ORDER BY state;

SELECT
    count(*) AS unfinished_live_requests
FROM execution_requests
WHERE requested_mode = 'LIVE'
  AND state IN (
      'READY',
      'CLAIMED',
      'SUBMITTING',
      'RECONCILING',
      'MANUAL_REVIEW_REQUIRED'
  );

SELECT
    count(*) AS zero_fill_active_positions
FROM trade_history
WHERE mode = 'LIVE'
  AND position_status IN (
      'OPENING',
      'OPEN',
      'PARTIALLY_CLOSED',
      'RESOLVED_REDEEMABLE'
  )
  AND COALESCE(entry_filled_shares, 0) = 0;
SQL
"""

cmd = ["ssh", "agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732", verify_script]
print('Running verification via SSH...')
result = subprocess.run(
    cmd, capture_output=True, text=True, encoding='utf-8'
)
print('STDOUT:')
print(result.stdout)
print('STDERR:')
print(result.stderr)
if result.returncode != 0:
    sys.exit(result.returncode)
