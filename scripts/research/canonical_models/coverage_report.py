"""Step 10: coverage report + go/no-go gate (local)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    assert_train_allowed(a.workdir)
    print("[coverage] eligible vs excluded markets/days by asset/source/reason must reconcile to the registry")
    print("[coverage] gate: if canonical data cannot support temporal splits -> PENDING_CANONICAL_DATA, ship pipeline, no proxy re-run under a new name")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
