"""Development CT-vs-control for one day-chunk (LOCAL ONLY).

Reads chunk files from --chunk-dir, writes scored markets + summary JSON to
--out-dir. CT history may reach into the previous day's depth chunk when
present (causal pre-history across day bounds). Registered PAPER decisions
are reconciled separately (--paper-csv optional) and never merged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from polyflip.research.canonical_models.dev_compare import (  # noqa: E402
    build_decisions, evaluate, summarize,
)
from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402
from polyflip.research.canonical_models.protocol import attach_hash  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--prev-day", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--paper-csv", default=None)
    a = ap.parse_args()
    assert_train_allowed(a.chunk_dir)
    cd, out = Path(a.chunk_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.day.replace("-", "")
    snap = cd / f"canonical_snap_{tag}.csv.gz"
    if not snap.exists():  # partial watermark chunk, e.g. 20260910p
        snap = cd / f"canonical_snap_{tag}p.csv.gz"
    snaps = pd.read_csv(snap, low_memory=False)
    depth_files = [cd / f"canonical_depth_{tag}.csv.gz",
                   cd / f"canonical_depth_{tag}p.csv.gz"]
    if a.prev_day:
        pt = a.prev_day.replace("-", "")
        depth_files += [cd / f"canonical_depth_{pt}.csv.gz",
                        cd / f"canonical_depth_{pt}p.csv.gz"]
    depth = pd.concat([pd.read_csv(f) for f in depth_files if f.exists()],
                      ignore_index=True)
    live = pd.read_csv(cd / "canonical_live_markets.csv.gz", low_memory=False)
    decisions = build_decisions(snaps, depth, live)
    scored = evaluate(decisions, depth)
    summary = attach_hash(summarize(scored))
    summary["day"] = a.day
    summary["excluded_reasons"] = (decisions.loc[decisions["status"] != "OK", "reason"]
                                   .value_counts().to_dict())
    summary["data_status_mix"] = (scored.loc[scored["in_sample"] == True, "data_status"]  # noqa: E712
                                  .value_counts().to_dict())
    scored.to_pickle(out / f"scored_{tag}.pkl")
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    if a.paper_csv and Path(a.paper_csv).exists():
        paper = pd.read_csv(a.paper_csv)
        mids = set(scored.loc[scored["in_sample"] == True, "market_id"].astype(str))  # noqa: E712
        pm = paper[paper["market_id"].astype(str).isin(mids)]
        recon = {"paper_rows_in_sample_markets": len(pm),
                 "note": "REGISTERED PAPER decisions kept separate; agreement counted, never merged"}
        (out / f"paper_recon_{tag}.json").write_text(json.dumps(recon, indent=1, default=str))
        print("paper_recon:", recon)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
