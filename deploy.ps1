# deploy.ps1 — Deployement script for Polyflip
# Usage: .\deploy.ps1 [-SkipTests]
param([switch]$SkipTests)

$ErrorActionPreference = "Stop"

if (-not $SkipTests) {
    Write-Host "Running pre-flight tests..." -ForegroundColor Cyan
    $env:PYTHONPATH = "."
    poetry run pytest tests/ -m "not live" -q --tb=short
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests failed. Aborting deploy." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Tests passed." -ForegroundColor Green
}

Write-Host "Step 1: Pushing to GitHub..." -ForegroundColor Cyan
git push origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "git push failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Pushed to GitHub." -ForegroundColor Green

Write-Host "Step 2: Deploying to server..." -ForegroundColor Cyan
ssh agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732 @'
set -e
cd polymarket-bot
git pull origin main

docker compose stop scheduler execution_worker_paper execution_worker_live execution_worker_shadow api || true
docker rm -f $(docker ps -aq --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME:-flipoly}" --filter "label=com.docker.compose.service=execution_worker") 2>/dev/null || true
docker rm -f $(docker ps -aq --filter "name=^/execution_worker$") 2>/dev/null || true
docker rm -f $(docker ps -aq --filter "name=^/polyflip_execution_worker$") 2>/dev/null || true

docker compose build
docker compose run --rm api alembic upgrade head

docker compose up -d --remove-orphans api scheduler execution_worker_paper
sleep 10
curl --fail http://localhost:8001/dashboard
'@
if ($LASTEXITCODE -ne 0) {
    Write-Host "Deploy failed on server." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Deploy complete!" -ForegroundColor Green
