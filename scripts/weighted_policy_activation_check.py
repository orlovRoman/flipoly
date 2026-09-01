#!/usr/bin/env python3
"""Evaluate the weighted-policy activation gate from explicit evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    )
    gate = activation_gate(evidence)
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
