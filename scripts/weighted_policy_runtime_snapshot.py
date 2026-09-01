#!/usr/bin/env python3
"""Write a redacted runtime snapshot for weighted-policy rollout audits."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_ENV_KEYS = (
    "POLYFLIP_BUILD_SHA",
    "TRADING_POLICY_MODE",
    "WEIGHTED_POLICY_ID",
    "WEIGHTED_MARKET_WEIGHT",
    "WEIGHTED_LOGREG_WEIGHT",
    "WEIGHTED_LGBM_WEIGHT",
    "WEIGHTED_MRF_BETA",
    "WEIGHTED_FEE_RATE",
    "WEIGHTED_MAKER_FEE_RATE",
    "WEIGHTED_FEE_EXPONENT",
    "WEIGHTED_SLIPPAGE_RATE",
    "WEIGHTED_LATENCY_BUFFER",
    "WEIGHTED_EXECUTION_ROLE",
    "WEIGHTED_MIN_NET_EV_FAVORITE",
    "WEIGHTED_MIN_NET_EV_OUTSIDER",
    "WEIGHTED_FIXED_BET_USDC",
    "WEIGHTED_MRF_EXTREME_VETO_THRESHOLD",
    "MARKET_REGIME_FILTER_MODE",
    "LIVE_TRADING_ENABLED",
    "EXECUTION_MODE",
)


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_snapshot() -> dict[str, Any]:
    environment = {
        key: os.getenv(key)
        for key in SAFE_ENV_KEYS
        if os.getenv(key) is not None
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "safe_environment": environment,
        "secrets_omitted": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--assert-live-disabled",
        action="store_true",
        help="fail if the snapshot says that live trading is enabled",
    )
    args = parser.parse_args()
    snapshot = build_snapshot()
    if args.assert_live_disabled:
        live_enabled = str(
            snapshot["safe_environment"].get("LIVE_TRADING_ENABLED", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if live_enabled:
            raise SystemExit("LIVE_TRADING_ENABLED is true; refusing rollout snapshot")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
