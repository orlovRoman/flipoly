import subprocess
import sys

def run_remote(cmd):
    res = subprocess.run(['ssh', 'agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732', cmd], capture_output=True, text=True, encoding='utf-8')
    return res

print("\n=== Check Database: Heartbeat ===")
res = run_remote('''sudo docker exec polyflip_db psql -U polyflip -d polyflip -c "SELECT * FROM health_status WHERE component = 'execution_worker_live';"''')
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

print("\n=== Check Database: Stuck Requests ===")
res = run_remote('''sudo docker exec polyflip_db psql -U polyflip -d polyflip -c "SELECT id, state, updated_at FROM execution_requests WHERE environment = 'live' AND state IN ('PENDING_EXECUTION', 'EXECUTING');"''')
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
