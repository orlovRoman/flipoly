import subprocess
import sys

def run_remote(cmd):
    res = subprocess.run(['ssh', 'agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732', cmd], capture_output=True, text=True, encoding='utf-8')
    return res

print("=== Check API Logs ===")
res = run_remote("cd /home/orlovrp/flipoly && sudo docker compose logs --tail=50 api | grep -iE 'Traceback|ERROR|500|exception|failed' || echo 'No errors'")
print(res.stdout, res.stderr)

print("\n=== Check Live Worker Logs ===")
res = run_remote("cd /home/orlovrp/flipoly && sudo docker compose --env-file /home/orlovrp/.flipoly-live-v2.env -f docker-compose.live-v2.yml --profile live-v2 logs --tail=50 execution_worker_live | grep -iE 'Traceback|ERROR|500|exception|failed' || echo 'No errors'")
print(res.stdout, res.stderr)

print("\n=== Check Database: Heartbeat ===")
res = run_remote('''sudo docker exec polyflip_db psql -U postgres -d polyflip -c "SELECT * FROM health_status WHERE component = 'execution_worker_live';"''')
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

print("\n=== Check Database: Stuck Requests ===")
res = run_remote('''sudo docker exec polyflip_db psql -U postgres -d polyflip -c "SELECT id, state, updated_at FROM execution_requests WHERE environment = 'live' AND state IN ('PENDING_EXECUTION', 'EXECUTING') AND updated_at < NOW() - INTERVAL '5 minutes';"''')
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

