#!/usr/bin/env python3
"""Evaluate the weighted-policy activation gate from explicit evidence."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
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
    parser.add_argument("--shadow-days", type=float)
    parser.add_argument("--shadow-resolved-markets", type=int)
    parser.add_argument("--shadow-candidate-trades", type=int)
    parser.add_argument("--repeat-oot-reports", type=int)
    parser.add_argument("--live-fills", type=int)
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
    parser.add_argument(
        "--require-live-validation",
        action="store_true",
        help="enforce T57 minimum LIVE fills and execution/calibration limits",
    )
    parser.add_argument("--artifact")
    parser.add_argument(
        "--policy-id",
        help="expected immutable policy ID when no artifact file is supplied",
    )
    parser.add_argument(
        "--evidence",
        help="load evidence values from weighted_policy_shadow_evidence.py output",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    evidence_values = {}
    raw_evidence: Mapping[str, object] = {}
    if args.evidence:
        loaded = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit("evidence file must contain an object")
        raw_evidence = loaded
        evidence_values = loaded.get("evidence", loaded)
        if not isinstance(evidence_values, dict):
            raise SystemExit("evidence file must contain an object or an evidence object")

    def value(name: str, default=None):
        explicit = getattr(args, name)
        if explicit is not None:
            return explicit
        return evidence_values.get(name, default)

    artifact_id = None
    if args.artifact:
        artifact = load_policy_artifact(Path(args.artifact))
        artifact_id = artifact.artifact_id

    def policy_ids_from(*sources: object) -> tuple[str, ...]:
        result: set[str] = set()
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            scalar = source.get("policy_id")
            if scalar is not None and str(scalar).strip():
                result.add(str(scalar).strip())
            values = source.get("policy_ids")
            if isinstance(values, (list, tuple, set)):
                result.update(str(item).strip() for item in values if str(item).strip())
        return tuple(sorted(result))

    observed_policy_ids = policy_ids_from(
        evidence_values,
        raw_evidence,
        raw_evidence.get("shadow"),
        raw_evidence.get("live"),
    )
    observed_policy_id = observed_policy_ids[0] if len(observed_policy_ids) == 1 else None
    evidence = ActivationEvidence(
        shadow_days=value("shadow_days", 0.0),
        shadow_resolved_markets=value("shadow_resolved_markets", 0),
        shadow_candidate_trades=value("shadow_candidate_trades", 0),
        repeat_oot_reports=value("repeat_oot_reports", 0),
        live_fills=value("live_fills", 0),
        pnl_ci_lower=value("pnl_ci_lower"),
        weighted_brier=value("weighted_brier"),
        market_brier=value("market_brier"),
        legacy_brier=value("legacy_brier"),
        weighted_net_pnl=value("weighted_net_pnl"),
        market_net_pnl=value("market_net_pnl"),
        legacy_net_pnl=value("legacy_net_pnl"),
        execution_drag=value("execution_drag"),
        calibration_error=value("calibration_error"),
        policy_id=observed_policy_id,
        policy_ids=observed_policy_ids,
    )
    expected_policy_id = artifact_id or (str(args.policy_id).strip() if args.policy_id else None)
    gate = activation_gate(
        evidence,
        min_brier_improvement=args.min_brier_improvement,
        max_execution_drag=args.max_execution_drag,
        max_calibration_error=args.max_calibration_error,
        require_live_validation=args.require_live_validation,
        expected_policy_id=expected_policy_id,
    )
    reasons = list(gate.reasons)
    if artifact_id and args.policy_id and str(args.policy_id).strip() != artifact_id:
        reasons.append("POLICY_ID_ARGUMENT_MISMATCH")
    payload = {
        "eligible": not reasons,
        "reasons": reasons,
        "evidence": evidence.__dict__,
        "artifact_id": artifact_id,
        "expected_policy_id": expected_policy_id,
        "evidence_policy_id": observed_policy_id,
        "evidence_policy_ids": list(observed_policy_ids),
        "evidence_source": args.evidence,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
