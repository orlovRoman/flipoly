#!/usr/bin/env bash
# ==============================================================================
# Script: run_continuous_shadow.sh
# Purpose: Continuous background shadow data collection for Polymarket LP Rewards (T03)
# Protocol: polymarket_lp_rewards_v0.1
#
# HARD SAFETY GATES:
#   - LP_LIVE_ENABLED MUST be "false" (or unset, defaults to false)
#   - Disk Watchdog halts collection automatically if free space drops below threshold (<20 GB)
#   - RotatingFileHandler limits logs to 20MB x 5 backups (max 100MB log footprint)
#   - Runs in passive shadow observation mode only. Zero CLOB post/orders sent.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# 1. Guarantee environment PATH & PYTHONPATH
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# 2. Enforce strict shadow mode
if [ "${LP_LIVE_ENABLED:-false}" = "true" ]; then
    echo "[FATAL] LP_LIVE_ENABLED is set to 'true'. Live trading is strictly prohibited before 7 full UTC days and Gate A passing." >&2
    exit 1
fi
export LP_LIVE_ENABLED="false"

# 3. Resolve storage root for Linux / cross-platform execution
DEFAULT_STORAGE_ROOT="${HOME}/flipoly-research/lp-rewards"
export LP_STORAGE_ROOT="${LP_STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"

# 4. Resolve Python runner / interpreter
PYTHON_CMD=()

if [ -n "${PYTHON_RUNNER:-}" ]; then
    # Custom runner override provided by caller
    # shellcheck disable=SC2206
    PYTHON_CMD=(${PYTHON_RUNNER})
elif command -v poetry >/dev/null 2>&1; then
    PYTHON_CMD=(poetry run python)
elif [ -x "${HOME}/.local/bin/poetry" ]; then
    PYTHON_CMD=("${HOME}/.local/bin/poetry" run python)
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
elif [ -x "${HOME}/.local/bin/uv" ]; then
    PYTHON_CMD=("${HOME}/.local/bin/uv" run python)
elif [ -x "${HOME}/.cargo/bin/uv" ]; then
    PYTHON_CMD=("${HOME}/.cargo/bin/uv" run python)
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_CMD=("${VIRTUAL_ENV}/bin/python")
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
elif [ -x "/usr/bin/python3" ]; then
    PYTHON_CMD=(/usr/bin/python3)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
else
    echo "[FATAL] No suitable Python interpreter or runner (poetry, uv, virtualenv, python3) found." >&2
    exit 1
fi

# 5. Setup log paths
LOG_DIR="${REPO_ROOT}/artifacts/research/lp_rewards/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/shadow_collector_continuous.log"

echo "=================================================================="
echo "Starting Continuous Shadow Data Collector (Protocol v0.1)"
echo "Mode: STRICT SHADOW (LP_LIVE_ENABLED=${LP_LIVE_ENABLED})"
echo "Working directory: ${REPO_ROOT}"
echo "Storage Root: ${LP_STORAGE_ROOT}"
echo "Interpreter / Runner: ${PYTHON_CMD[*]}"
echo "Log file: ${LOG_FILE} (20MB rotating, 5 backups)"
echo "Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================================="

# 6. Storage arguments handling
STORAGE_ARGS=()
HAS_STORAGE_ARG=0
for arg in "$@"; do
    if [[ "${arg}" == "--storage-root"* ]]; then
        HAS_STORAGE_ARG=1
        break
    fi
done

if [ "${HAS_STORAGE_ARG}" -eq 0 ] && [ -n "${LP_STORAGE_ROOT:-}" ]; then
    STORAGE_ARGS=(--storage-root "${LP_STORAGE_ROOT}")
fi

# 7. Launch shadow collector
# In background via nohup:
#   nohup ./scripts/research/lp_rewards/run_continuous_shadow.sh > /dev/null 2>&1 &
if [ "${#STORAGE_ARGS[@]}" -gt 0 ]; then
    exec "${PYTHON_CMD[@]}" -u "${SCRIPT_DIR}/03_run_shadow_collector.py" --log-file "${LOG_FILE}" "${STORAGE_ARGS[@]}" "$@"
else
    exec "${PYTHON_CMD[@]}" -u "${SCRIPT_DIR}/03_run_shadow_collector.py" --log-file "${LOG_FILE}" "$@"
fi
