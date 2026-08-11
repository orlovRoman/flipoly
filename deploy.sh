#!/usr/bin/env bash
# Usage:
#   ./deploy.sh           — обычный деплой (использует кэш Docker, ~1-2 мин)
#   ./deploy.sh --no-cache — полная пересборка с нуля (~15-20 мин, нужно только при изменении requirements.txt)
set -e

ENV_FILE="/home/orlovrp/.flipoly-live-v2.env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE=".env"
fi

BUILD_ARGS=""
if [[ "$1" == "--no-cache" ]]; then
    BUILD_ARGS="--no-cache"
    echo "=== 1. Building all services from scratch (NO CACHE) ==="
else
    echo "=== 1. Building all services (with cache) ==="
fi

docker compose --env-file "$ENV_FILE" --profile live-v2 build $BUILD_ARGS

echo "=== 2. Recreating all containers (Core + Live-v2) ==="
docker compose --env-file "$ENV_FILE" --profile live-v2 up -d --force-recreate

echo "=== 3. Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}"
