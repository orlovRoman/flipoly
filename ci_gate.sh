#!/bin/bash
set -e

echo "=== Running Pytest ==="
python -m pytest

echo "=== Running Execution Worker (Dry-Run) ==="
python -m polyflip.execution.worker --dry-run

echo ""
echo -e "\033[42;37m [ SUCCESS ] READY FOR LIVE \033[0m"
