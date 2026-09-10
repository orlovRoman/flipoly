"""PAPER-moment comparison runner (LOCAL ONLY). See paper_moment module."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402
from polyflip.research.canonical_models.paper_moment import run, summarize  # noqa: E402
from polyflip.research.canonical_models.protocol import attach_hash  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    assert_train_allowed(a.chunk_dir)
    cd, out = Path(a.chunk_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paper = pd.read_csv(cd / "canonical_ct_reservations.csv.gz")
    paper["decision_at"] = pd.to_datetime(paper["decision_at"], utc=True, format="mixed")
    lo = (paper["decision_at"].min() - pd.Timedelta(days=1)).date().isoformat()
    hi = paper["decision_at"].max().date().isoformat()
    # Load only chunks overlapping the reservation window (chunked discipline;
    # a full-history concat blows memory).
    day_tags = []
    d0 = pd.Timestamp(lo).date()
    d1 = pd.Timestamp(hi).date()
    while d0 <= d1:
        day_tags.append(d0.strftime("%Y%m%d"))
        d0 += pd.Timedelta(days=1)
    snap_usecols = ["market_id", "recorded_at", "mid_price", "best_bid",
                    "best_ask", "final_outcome"]
    snaps = pd.concat(
        [pd.read_csv(cd / f"canonical_snap_{t}.csv.gz", low_memory=False,
                     usecols=lambda c: c in snap_usecols)
         for t in day_tags if (cd / f"canonical_snap_{t}.csv.gz").exists()]
        + [pd.read_csv(cd / f"canonical_snap_{t}p.csv.gz", low_memory=False,
                       usecols=lambda c: c in snap_usecols)
           for t in day_tags if (cd / f"canonical_snap_{t}p.csv.gz").exists()],
        ignore_index=True)
    depth_files = [cd / f"canonical_depth_{t}.csv.gz" for t in day_tags]
    depth_files += [cd / f"canonical_depth_{t}p.csv.gz" for t in day_tags]
    depth = pd.concat([pd.read_csv(f) for f in depth_files if f.exists()],
                      ignore_index=True)
    live = pd.read_csv(cd / "canonical_live_markets.csv.gz", low_memory=False)
    df = run(paper, snaps, depth, live)
    summary = attach_hash(summarize(df))
    df.to_pickle(out / "paper_moment_scored.pkl")
    (out / "paper_moment_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
