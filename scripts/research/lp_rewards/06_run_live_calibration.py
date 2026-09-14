"""06_run_live_calibration.py

Entry point for live calibration on $100 working capital (Days 8-21).
Protected by LP_LIVE_ENABLED hard gate and protocol SHA-256 verification.
"""

import os
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.execution import LiveOrderExecutor
from polyflip.research.lp_rewards.protocol import load_protocol


def main():
    protocol = load_protocol()
    print(f"Protocol loaded: {protocol.protocol_id} (SHA-256: {protocol.sha256_hash})")

    executor = LiveOrderExecutor(expected_protocol_hash=protocol.sha256_hash)
    if not executor.is_live_enabled():
        print("\n[BLOCKED BY HARD GATE] LP_LIVE_ENABLED is not 'true'.")
        print("Live orders cannot be placed until Gate A is passed and LP_LIVE_ENABLED=true is explicitly set in the environment.")
        sys.exit(1)

    print("\n[GATE PASSED] LP_LIVE_ENABLED=true. Live calibration mode is active.")


if __name__ == "__main__":
    main()
