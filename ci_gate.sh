#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Python compilation ==="
python -m compileall -q polyflip

echo "=== Checking dashboard JavaScript ==="
node --check polyflip/static/js/app.js

echo "=== Checking Alembic migrations ==="
python -m alembic heads
HEAD_COUNT=$(python -m alembic heads | wc -l)
if [ "$HEAD_COUNT" -ne 1 ]; then
    echo "ERROR: Expected 1 alembic head, found $HEAD_COUNT" >&2
    exit 1
fi

echo "=== Running tests ==="
python -m pytest tests/ -m "not live" --strict-config --strict-markers -q

echo "=== Checking code style (execution contour) ==="
python -m flake8 \
    polyflip/execution \
    polyflip/trading/stoploss_worker.py \
    polyflip/trading/takeprofit_worker.py \
    --max-line-length=100

python -m black --check \
    polyflip/execution \
    polyflip/trading/stoploss_worker.py \
    polyflip/trading/takeprofit_worker.py

echo "=== Checking for trailing whitespace ==="
git diff --check HEAD~1 HEAD 2>/dev/null || true

echo "=== CI gate passed! ==="
