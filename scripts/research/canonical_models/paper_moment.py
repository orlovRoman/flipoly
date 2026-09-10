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
    snaps = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in sorted(cd.glob("canonical_snap_*.csv.gz"))],
        ignore_index=True)
    depth = pd.concat(
        [pd.read_csv(f) for f in sorted(cd.glob("canonical_depth_*.csv.gz"))],
        ignore_index=True)
    live = pd.read_csv(cd / "canonical_live_markets.csv.gz", low_memory=False)
    paper = pd.read_csv(cd / "canonical_ct_reservations.csv.gz")
    df = run(paper, snaps, depth, live)
    summary = attach_hash(summarize(df))
    df.to_pickle(out / "paper_moment_scored.pkl")
    (out / "paper_moment_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
