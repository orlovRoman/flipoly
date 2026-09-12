"""Common-row comparison of the 3-feature LogReg family (LOCAL ONLY, chunked).

Period, group and policy are fixed here (see legacy_compare module doc).
Writes scored markets + forecast/policy tables to --out-dir.
v22 (26-feature pipeline) stays PENDING_FEATURE_MAP.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from polyflip.research.legacy_btc_logreg.legacy_compare import (  # noqa: E402
    apply_policy, build_opportunities, forecast_table,
)

GROUP_IDS = [827, 828, 838, 839, 840, 850, 851, 852, 862, 863, 864,
             874, 875, 876]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--blob-csv", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    _wd = str(Path(a.chunk_dir).resolve()).lower().replace("/", "\\")
    assert "sshfs" not in _wd and not _wd.startswith("\\\\"), \
        "research runs on a local disk only"
    cd, out = Path(a.chunk_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blobs = pd.read_csv(a.blob_csv)
    arts = {}
    for rid in GROUP_IDS:
        hx = blobs[blobs.id == rid].blob_hex.iloc[0]
        raw = bytes.fromhex(hx)
        arts[rid] = (raw, "sha256:" + hashlib.sha256(raw).hexdigest())
    live = pd.read_csv(cd / "canonical_live_markets.csv.gz", low_memory=False)
    frames = []
    for f in sorted(glob.glob(str(cd / "canonical_snap_*.csv.gz"))):
        day = Path(f).stem.split("_")[2][:8]
        if not (a.start.replace("-", "") <= day <= a.end.replace("-", "")):
            continue
        frames.append(pd.read_csv(
            f, low_memory=False,
            usecols=lambda c: c in ("market_id", "recorded_at", "mid_price",
                                    "spread", "best_ask", "best_bid", "final_outcome")))
    opps = build_opportunities(pd.concat(frames, ignore_index=True), live)
    opps["p_win_placeholder"] = 0.0
    import polyflip.research.legacy_btc_logreg.legacy_compare as LC
    df = opps.copy()
    rows = df[LC.FEATURES_3].to_dict("records")
    import math
    from polyflip.research.legacy_btc_logreg.pinned_loader import load_pinned
    for rid, (blob, sha) in sorted(arts.items()):
        model, order = load_pinned(blob, sha, LC.FEATURES_3)
        coef, b = model.coef_[0], float(model.intercept_[0])
        df[f"p_flip_{rid}"] = [1.0 / (1.0 + math.exp(-(b + sum(
            c * r[f] for c, f in zip(coef, order))))) for r in rows]
    res = {"n": len(df), "period": [a.start, a.end],
           "by_asset": df["asset"].value_counts().to_dict(),
           "forecast": forecast_table(df, GROUP_IDS), "policy": {}}
    for rid in GROUP_IDS:
        pol = apply_policy(df, rid)
        e = pol[pol["enter"]]
        res["policy"][str(rid)] = {
            "entries": int(e.shape[0]),
            "gross": round(float(e["gross"].sum()), 4),
            "win_rate": round(float((e["payout"] > 0).mean()), 4) if len(e) else 0.0,
            "fee_0.002": round(float(e["fee_0.002"].sum()), 4),
        }
    df.to_pickle(out / "scored_common.pkl")
    (out / "common_tables.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"n": res["n"], "by_asset": res["by_asset"]}, indent=1))
    for rid in GROUP_IDS:
        f = res["forecast"][str(rid)]
        p = res["policy"][str(rid)]
        print(rid, "brier", f["brier"], "logloss", f["logloss"],
              "| entries", p["entries"], "gross", p["gross"], "wr", p["win_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
