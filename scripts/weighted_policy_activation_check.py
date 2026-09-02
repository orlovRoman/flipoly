#!/usr/bin/env python3
"""Evaluate the weighted-policy activation gate from explicit evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyflip.trading.policy_artifact import (
    ActivationEvidence,
    activation_gate,
    load_policy_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-days", type=float, required=True)
    parser.add_argument("--shadow-resolved-markets", type=int, required=True)
    parser.add_argument("--shadow-candidate-trades", type=int, required=True)
    parser.add_argument("--repeat-oot-reports", type=int, required=True)
    parser.add_argument("--live-fills", type=int, required=True)
    parser.add_argument("--pnl-ci-lower", type=float)
    parser.add_argument("--weighted-brier", type=float)
    parser.add_argument("--market-brier", type=float)
    parser.add_argument("--legacy-brier", type=float)
    parser.add_argument("--weighted-net-pnl", type=float)
    parser.add_argument("--market-net-pnl", type=float)
    parser.add_argument("--legacy-net-pnl", type=float)
    parser.add_argument("--execution-drag", type=float)
    parser.add_argument("--calibration-error", type=float)
    parser.add_argument("--min-brier-improvement", type=float, default=0.002)
    parser.add_argument("--max-execution-drag", type=float, default=0.02)
    parser.add_argument("--max-calibration-error", type=float, default=0.05)
    parser.add_argument("--artifact")
    parser.add_argument("--output")
    args = parser.parse_args()

    artifact_id = None
    if args.artifact:
        artifact = load_policy_artifact(Path(args.artifact))
        artifact_id = artifact.artifact_id
    evidence = ActivationEvidence(
        shadow_days=args.shadow_days,
        shadow_resolved_markets=args.shadow_resolved_markets,
        shadow_candidate_trades=args.shadow_candidate_trades,
        repeat_oot_reports=args.repeat_oot_reports,
        live_fills=args.live_fills,
        pnl_ci_lower=args.pnl_ci_lower,
        weighted_brier=args.weighted_brier,
        market_brier=args.market_brier,
        legacy_brier=args.legacy_brier,
        weighted_net_pnl=args.weighted_net_pnl,
        market_net_pnl=args.market_net_pnl,
        legacy_net_pnl=args.legacy_net_pnl,
        execution_drag=args.execution_drag,
        calibration_error=args.calibration_error,
    )
    gate = activation_gate(
        evidence,
        min_brier_improvement=args.min_brier_improvement,
        max_execution_drag=args.max_execution_drag,
        max_calibration_error=args.max_calibration_error,
    )
    payload = {
        "eligible": gate.eligible,
        "reasons": list(gate.reasons),
        "evidence": evidence.__dict__,
        "artifact_id": artifact_id,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if gate.eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
