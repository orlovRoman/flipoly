"""Single entry-point for the canonical comparison (local only, no scheduler/worker)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402
from polyflip.research.canonical_models.manifest import write_run_manifest  # noqa: E402
from polyflip.research.canonical_models.protocol import load_protocol, protocol_hash  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true", help="synthetic tiny run, no DB")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    root = assert_train_allowed(a.workdir)
    proto = load_protocol()
    print(f"[canonical] protocol={proto['protocol_version']} hash={protocol_hash(proto)[:12]}")
    print(f"[canonical] workdir={root} run={a.run_id} smoke={a.smoke} resume={a.resume}")
    print("[canonical] stages: audit -> export-check -> registry -> dataset -> train -> forecast -> bootstrap -> economics -> report")
    print("[canonical] (smoke mode validates wiring only; real run needs coverage-gated periods)")
    write_run_manifest(REPO, a.run_id, {"mode": "smoke" if a.smoke else "full",
                                       "protocol_hash": protocol_hash(proto)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
