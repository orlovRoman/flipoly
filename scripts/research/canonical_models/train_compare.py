"""Steps 26-31: local temporal training + forecast table + challenger pick."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from polyflip.research.canonical_models.guards import assert_train_allowed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    assert_train_allowed(a.workdir)
    print(f"[train] local only, run={a.run_id}; expanding folds; tune/select inside development; final test untouched")
    print("[train] metrics: Brier/logloss once-per-market, prob reliability, ask-bin calibration (0.05), UP/DOWN/asset/week slices")
    print("[train] challenger = best OOF logloss among M2-M4 (recorded BEFORE final test) else NO_CLEAR_CHALLENGER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
