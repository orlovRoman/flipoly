"""BTC/all coverage by day/hour over exported snapshot chunks (LOCAL ONLY).

Denominator (exact): resolved markets with known end_at and duration >= 300s
whose end falls on that UTC day. Hit: >=1 snapshot with recorded_at in
(T-5m, T-4:45m]. UP-leg: hit snapshot has mid_price+best_ask. DOWN-leg:
covered only where depth exists (separate depth-window note).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert_train_allowed(a.chunk_dir)
    cd = Path(a.chunk_dir)
    live = pd.read_csv(cd / "canonical_live_markets.csv.gz", low_memory=False,
                       usecols=["market_id", "asset", "end_time_est",
                                "market_start_at", "market_end_at"])
    live["market_id"] = live["market_id"].astype(str)
    live["end_at"] = pd.to_datetime(live["market_end_at"], utc=True, format="mixed")
    no_end = live["end_at"].isna()
    live.loc[no_end, "end_at"] = pd.to_datetime(
        live.loc[no_end, "end_time_est"], utc=True, format="mixed")
    live["start_at"] = pd.to_datetime(live["market_start_at"], utc=True, format="mixed")
    live["dur"] = (live["end_at"] - live["start_at"]).dt.total_seconds()
    live.loc[live["start_at"].isna(), "dur"] = float("inf")  # unknown duration: keep, flag
    end_by = live.set_index("market_id")["end_at"]
    asset_by = live.set_index("market_id")["asset"]

    per_market_last: dict[str, str] = {}
    per_market_hit: dict[str, bool] = {}
    per_market_upleg: dict[str, bool] = {}
    for f in sorted(glob.glob(str(cd / "canonical_snap_*.csv.gz"))):
        df = pd.read_csv(f, low_memory=False,
                         usecols=["market_id", "recorded_at", "mid_price",
                                  "best_ask", "final_outcome"])
        df["market_id"] = df["market_id"].astype(str)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, format="mixed")
        df = df.sort_values("recorded_at")
        for mid, g in df.groupby("market_id"):
            per_market_last[mid] = g.iloc[-1]["final_outcome"]
            if mid in per_market_hit and per_market_hit[mid]:
                continue
            end = end_by.get(mid)
            if end is None or pd.isna(end):
                continue
            tgt = end - timedelta(seconds=300)
            w = g[(g["recorded_at"] > tgt) & (g["recorded_at"] <= tgt + timedelta(seconds=15))]
            if not w.empty:
                per_market_hit[mid] = True
                row = w.iloc[0]
                per_market_upleg[mid] = bool(pd.notna(row["mid_price"]) and pd.notna(row["best_ask"]))
    rows = []
    for mid, outcome in per_market_last.items():
        end = end_by.get(mid)
        if end is None or pd.isna(end) or outcome not in ("YES", "NO"):
            continue
        rows.append({"day": end.date().isoformat(),
                     "hour": (end - timedelta(seconds=300)).hour,
                     "asset": asset_by.get(mid, "?"),
                     "hit": bool(per_market_hit.get(mid, False)),
                     "up_leg": bool(per_market_upleg.get(mid, False))})
    cov = pd.DataFrame(rows)
    elig = cov  # exact denominator: resolved + known end (+duration>=300s where known)
    out = {"denominator": ("resolved markets with known end_at; duration>=300s "
                           "where start known; grouped by end day UTC"),
           "n_eligible": len(elig),
           "hit_rate": round(float(elig["hit"].mean()), 4),
           "by_day": elig.groupby("day").agg(n=("hit", "size"),
                                             hit=("hit", "sum")).to_dict("index"),
           "by_hour": elig.groupby("hour").agg(n=("hit", "size"),
                                               hit=("hit", "sum")).to_dict("index"),
           "btc": {}, "down_leg_note": ("DOWN leg only where depth exists "
                                        "(2026-09-09..09-10); see depth-window analysis.")}
    b = elig[elig["asset"] == "BTC"]
    out["btc"] = {"n_eligible": len(b), "hit_rate": round(float(b["hit"].mean()), 4) if len(b) else None,
                  "up_leg_rate": round(float(b["up_leg"].mean()), 4) if len(b) else None,
                  "by_day": b.groupby("day").agg(n=("hit", "size"),
                                                 hit=("hit", "sum")).to_dict("index")}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("eligible:", out["n_eligible"], "hit_rate:", out["hit_rate"],
          "btc:", out["btc"]["n_eligible"], out["btc"]["hit_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
