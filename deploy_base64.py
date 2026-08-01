import base64
import subprocess
import sys

deploy_script = """cd /home/orlovrp/flipoly
set -Eeuo pipefail

test -z "$(git status --porcelain)" || {
  echo "ОШИБКА: есть локальные изменения"
  git status --short
  exit 1
}

git switch main
git fetch origin main
git merge --ff-only origin/main

echo "Развёртываем коммит:"
git rev-parse --short HEAD

docker compose build api

docker compose up -d \\
  --force-recreate \\
  api

docker compose \\
  --env-file /home/orlovrp/.flipoly-live-v2.env \\
  -f docker-compose.live-v2.yml \\
  --profile live-v2 \\
  up -d \\
  --force-recreate \\
  execution_worker_live \\
  live_mirror_worker \\
  release_gate_worker
""".encode('utf-8')

b64_script = base64.b64encode(deploy_script).decode('ascii')
cmd = f"echo '{b64_script}' | base64 -d > /tmp/deploy_live.sh && bash /tmp/deploy_live.sh"
print('Running deploy via SSH with base64...')

result = subprocess.run(
    ['ssh', 'agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732', cmd],
    capture_output=True, text=True, encoding='utf-8'
)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
if result.returncode != 0:
    sys.exit(result.returncode)
