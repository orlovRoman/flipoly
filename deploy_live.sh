cd /home/orlovrp/flipoly
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

docker compose up -d \
  --force-recreate \
  api

docker compose \
  --env-file /home/orlovrp/.flipoly-live-v2.env \
  -f docker-compose.live-v2.yml \
  --profile live-v2 \
  up -d \
  --force-recreate \
  execution_worker_live \
  live_mirror_worker \
  release_gate_worker
