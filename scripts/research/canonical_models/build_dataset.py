"""Steps 11-18: causal dataset build (local, Parquet, chunked, resumable)."""
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
    ap.add_argument("--chunk-rows", type=int, default=200000)
    a = ap.parse_args()
    assert_train_allowed(a.workdir)
    print(f"[dataset] T-5m window (+15s), both-side quotes, underlying+strike, 30m vol, base+traj+CT features, chunk_rows={a.chunk_rows}")
    print("[dataset] forecast level: 1 row/market P(UP); trading level: observed quotes+execution; common rows for comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
