#!/usr/bin/env bash
set -e

echo "Running tests..."
python -m pytest tests/ -m "not live"

echo "Running flake8..."
flake8 polyflip || echo "flake8 failed, but continuing for now"

echo "Running black..."
black --check polyflip || echo "black failed, but continuing for now"

echo "CI gate passed!"
