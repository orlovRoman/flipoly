"""Readiness gate for final activation (LOCAL ONLY).

For the latest N PAPER reservations, verify the full record cycle exists
in the local chunks: a depth snapshot of the decided token within 15 s of
decision_at AND a snapshot row within 60 s AND a known outcome.
Prints PASS/FAIL per reservation. Run AFTER frozen-code activation, BEFORE
setting final_start. Exit code 0 only if all checked rows pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--paper-csv", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    assert_train_allowed(a.chunk_dir)
    cd = Path(a.chunk_dir)
    depth = pd.concat([pd.read_csv(f) for f in sorted(cd.glob("canonical_depth_*.csv.gz"))],
                      ignore_index=True)
    snaps = pd.concat(
        [pd.read_csv(f, low_memory=False, usecols=["market_id", "recorded_at", "final_outcome"])
         for f in sorted(cd.glob("canonical_snap_*.csv.gz"))], ignore_index=True)
    for c in ("event_at", "received_at"):
        depth[c] = pd.to_datetime(depth[c], utc=True, format="mixed")
    depth["market_id"] = depth["market_id"].astype(str)
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"], utc=True, format="mixed")
    snaps["market_id"] = snaps["market_id"].astype(str)
    paper = pd.read_csv(a.paper_csv)
    paper["decision_at"] = pd.to_datetime(paper["decision_at"], utc=True, format="mixed")
    paper = paper.sort_values("decision_at").tail(a.n)
    fails = 0
    for _, r in paper.iterrows():
        mid, dec = str(r["market_id"]), r["decision_at"]
        d = depth[(depth["market_id"] == mid)
                  & (abs(depth["received_at"] - dec) <= pd.Timedelta(seconds=15))]
        s = snaps[(snaps["market_id"] == mid)
                  & (abs(snaps["recorded_at"] - dec) <= pd.Timedelta(seconds=60))]
        ok = (not d.empty) and (not s.empty) and (s.iloc[-1]["final_outcome"] in ("YES", "NO"))
        print(("PASS" if ok else "FAIL"), mid, str(dec),
              f"depth15s={len(d)} snap60s={len(s)}")
        fails += 0 if ok else 1
    print("readiness:", "PASS" if fails == 0 else f"FAIL({fails})")
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
