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

# 1. Enforce strict shadow mode
if [ "${LP_LIVE_ENABLED:-false}" = "true" ]; then
    echo "[FATAL] LP_LIVE_ENABLED is set to 'true'. Live trading is strictly prohibited before 7 full UTC days and Gate A passing." >&2
    exit 1
fi
export LP_LIVE_ENABLED="false"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# 2. Setup log paths
LOG_DIR="${REPO_ROOT}/artifacts/research/lp_rewards/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/shadow_collector_continuous.log"

echo "=================================================================="
echo "Starting Continuous Shadow Data Collector (Protocol v0.1)"
echo "Mode: STRICT SHADOW (LP_LIVE_ENABLED=${LP_LIVE_ENABLED})"
echo "Working directory: ${REPO_ROOT}"
echo "Log file: ${LOG_FILE} (20MB rotating, 5 backups)"
echo "Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================================="

# 3. Launch shadow collector
# In background via nohup:
#   nohup ./scripts/research/lp_rewards/run_continuous_shadow.sh > /dev/null 2>&1 &
exec python -u "${SCRIPT_DIR}/03_run_shadow_collector.py" --log-file "${LOG_FILE}" "$@"
