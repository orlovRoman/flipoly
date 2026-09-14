"""Continuous background shadow collector launcher for LP Rewards research.

Enforces:
  - LP_LIVE_ENABLED=false
  - Continuous execution with watchdog disk monitoring
  - Rotating file logging (20MB x 5 backups)
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Guard: fail-closed against live execution
    live_env = os.getenv("LP_LIVE_ENABLED", "false").lower()
    if live_env in ("true", "1", "yes"):
        print("[FATAL ERROR] LP_LIVE_ENABLED is set to true. Live execution is forbidden before Gate A approval.", file=sys.stderr)
        sys.exit(1)

    os.environ["LP_LIVE_ENABLED"] = "false"

    log_dir = repo_root / "artifacts" / "research" / "lp_rewards" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "shadow_collector_continuous.log"

    print("=" * 66)
    print("Launching continuous shadow data collector...")
    print(f"Log path: {log_file}")
    print("LP_LIVE_ENABLED: false (ENFORCED)")
    print("=" * 66)

    script_path = repo_root / "scripts" / "research" / "lp_rewards" / "03_run_shadow_collector.py"
    cmd = [sys.executable, "-u", str(script_path), "--log-file", str(log_file)] + sys.argv[1:]

    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
