import subprocess
import sys

deploy_script = """cd /home/orlovrp/flipoly
sudo chown orlovrp:orlovrp .git/index .git/ORIG_HEAD || true
sudo -u orlovrp git restore poetry.lock || true
sudo -u orlovrp git switch main
sudo -u orlovrp git fetch origin main
sudo -u orlovrp git merge --ff-only origin/main

echo "Развёртываем коммит:"
sudo -u orlovrp git rev-parse --short HEAD

sudo docker compose build api

sudo docker compose up -d \\
  --force-recreate \\
  api

sudo docker compose \\
  --env-file /home/orlovrp/.flipoly-live-v2.env \\
  -f docker-compose.live-v2.yml \\
  --profile live-v2 \\
  up -d \\
  --force-recreate \\
  execution_worker_live \\
  live_mirror_worker \\
  release_gate_worker
"""

cmd = ["ssh", "agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732", deploy_script]
print('Running deploy via SSH directly...')

result = subprocess.run(
    cmd, capture_output=True, text=True, encoding='utf-8'
)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
if result.returncode != 0:
    sys.exit(result.returncode)
