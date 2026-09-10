"""Steps 38-40: single report + computed status + reproducible manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from polyflip.research.canonical_models.manifest import write_run_manifest
from polyflip.research.canonical_models.protocol import attach_hash
from polyflip.research.canonical_models.report import build_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--summary", default="{}")
    a = ap.parse_args()
    rep = build_report({"summary": json.loads(a.summary)})
    rep = attach_hash(rep)
    out = REPO / "artifacts" / "research" / "canonical_models" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    write_run_manifest(REPO, a.run_id, {"report_status": rep["final_status"],
                                       "protocol_hash": rep["protocol_hash"]})
    print(f"[report] status={rep['final_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
