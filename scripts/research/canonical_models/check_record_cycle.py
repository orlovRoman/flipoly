"""Readiness gate for final activation (LOCAL ONLY).

For the latest N PAPER reservations, verify the full record cycle exists
in the local chunks: input_bundle present AND a depth snapshot of the decided
token within 15 s of decision_at AND a snapshot row within 60 s.
Outcome is reported (PENDING allowed) unless --require-outcome is passed.
Only day-chunks overlapping the checked decisions are loaded (chunked).
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
    ap.add_argument("--require-outcome", action="store_true")
    a = ap.parse_args()
    assert_train_allowed(a.chunk_dir)
    cd = Path(a.chunk_dir)
    paper_all = pd.read_csv(a.paper_csv)
    paper_all["decision_at"] = pd.to_datetime(paper_all["decision_at"], utc=True,
                                              format="mixed")
    paper = paper_all.sort_values("decision_at").tail(a.n)
    lo = (paper["decision_at"].min() - pd.Timedelta(days=1)).date().isoformat()
    hi = paper["decision_at"].max().date().isoformat()
    day_tags = []
    d0, d1 = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    while d0 <= d1:
        day_tags.append(d0.strftime("%Y%m%d"))
        d0 += pd.Timedelta(days=1)
    depth_files = [cd / f"canonical_depth_{t}.csv.gz" for t in day_tags]
    depth_files += [cd / f"canonical_depth_{t}p.csv.gz" for t in day_tags]
    depth = pd.concat([pd.read_csv(f) for f in depth_files if f.exists()],
                      ignore_index=True)
    snap_files = [cd / f"canonical_snap_{t}.csv.gz" for t in day_tags]
    snap_files += [cd / f"canonical_snap_{t}p.csv.gz" for t in day_tags]
    snaps = pd.concat(
        [pd.read_csv(f, low_memory=False,
                     usecols=lambda c: c in ("market_id", "recorded_at", "final_outcome"))
         for f in snap_files if f.exists()], ignore_index=True)
    for c in ("event_at", "received_at"):
        depth[c] = pd.to_datetime(depth[c], utc=True, format="mixed")
    depth["market_id"] = depth["market_id"].astype(str)
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"], utc=True, format="mixed")
    snaps["market_id"] = snaps["market_id"].astype(str)
    fails = 0
    for _, r in paper.iterrows():
        mid, dec = str(r["market_id"]), r["decision_at"]
        import json as _json
        det = r.get("decision_details")
        try:
            det = _json.loads(det) if isinstance(det, str) else (det or {})
        except Exception:
            det = {}
        bundle = det.get("input_bundle") or {}
        has_bundle = bool(bundle.get("up_history") or bundle.get("down_history")
                          or bundle.get("quotes"))
        d = depth[(depth["market_id"].astype(str) == mid)
                  & (abs(depth["received_at"] - dec) <= pd.Timedelta(seconds=15))]
        s = snaps[(snaps["market_id"].astype(str) == mid)
                  & (abs(snaps["recorded_at"] - dec) <= pd.Timedelta(seconds=60))]
        outcome = s.iloc[-1]["final_outcome"] if not s.empty else None
        outcome_ok = (outcome in ("YES", "NO")) if a.require_outcome else True
        ok = has_bundle and (not d.empty) and (not s.empty) and outcome_ok
        print(("PASS" if ok else "FAIL"), mid, str(dec),
              f"bundle={has_bundle} depth15s={len(d)} snap60s={len(s)} "
              f"outcome={outcome}")
        fails += 0 if ok else 1
    print("readiness:", "PASS" if fails == 0 else f"FAIL({fails})")
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
