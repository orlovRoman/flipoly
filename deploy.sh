#!/usr/bin/env bash
set -e

ENV_FILE="/home/orlovrp/.flipoly-live-v2.env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE=".env"
fi

echo "=== 1. Building all services from scratch (no-cache) ==="
docker compose --env-file "$ENV_FILE" --profile live-v2 build --no-cache

echo "=== 2. Recreating all containers (Core + Live-v2) ==="
docker compose --env-file "$ENV_FILE" --profile live-v2 up -d --force-recreate

echo "=== 3. Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}"
