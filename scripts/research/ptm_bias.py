"""PTM items 16-17: independent universe composition + funnel bias comparison.

Universe: expirations markets with snapshots in period, asset in 5, outcome
present, contract duration check. Compares funnel vs non-funnel groups.
Usage: python3 ptm_bias.py <freeze_name> <period_start_iso> <period_end_iso>
"""
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

FREEZE = sys.argv[1]
P0 = datetime.fromisoformat(sys.argv[2])
P1 = datetime.fromisoformat(sys.argv[3])
ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE")

fbase = os.path.join("artifacts", "research", "price_time_map", "_freeze", FREEZE)
exp = json.load(open("artifacts/research/market_expirations.json", encoding="utf-8"))
obs_doc = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))
funnel_mids = {str(o["market_id"]) for o in obs_doc["observations"] if o.get("asset") in ASSETS}

meta = json.load(open(os.path.join(fbase, "meta.json"), encoding="utf-8"))
markets = {}
for f in meta["files"]:
    with gzip.open(os.path.join(fbase, f["path"]), "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            m = markets.setdefault(r["market_id"], {"n": 0, "assets": Counter(), "outs": set(),
                                                    "first": None, "last": None})
            m["n"] += 1
            m["assets"][r.get("asset") or ""] += 1
            if r["final_outcome"]:
                m["outs"].add(r["final_outcome"])
            ts = r["recorded_at"]
            if m["first"] is None or ts < m["first"]:
                m["first"] = ts
            if m["last"] is None or ts > m["last"]:
                m["last"] = ts

rows = []
for mid, m in markets.items():
    if mid not in exp:
        continue
    expiry = datetime.fromisoformat(exp[mid])
    if not (P0 <= expiry <= P1):
        continue
    asset = m["assets"].most_common(1)[0][0]
    if asset not in ASSETS:
        continue
    first = datetime.fromisoformat(m["first"])
    duration_min = (expiry - first).total_seconds() / 60.0
    rows.append({"market_id": mid, "asset": asset, "expiry": exp[mid],
                 "duration_min": round(duration_min, 1), "snapshots": m["n"],
                 "outcome": sorted(m["outs"])[0] if len(m["outs"]) == 1 else ("CONFLICT" if m["outs"] else "MISSING"),
                 "in_funnel": mid in funnel_mids})

json.dump({"period": [sys.argv[2], sys.argv[3]], "markets": rows},
          open(os.path.join(fbase, "..", "universe_%s.json" % FREEZE), "w"), indent=1)
in_f = [r for r in rows if r["in_funnel"]]
out_f = [r for r in rows if not r["in_funnel"]]
print("universe markets:", len(rows), "funnel:", len(in_f), "non-funnel:", len(out_f))


def desc(rs):
    d = {"n": len(rs), "assets": dict(Counter(r["asset"] for r in rs)),
         "outcomes": dict(Counter(r["outcome"] for r in rs))}
    ds = sorted({r["expiry"][:10] for r in rs})
    d["days"] = len(ds)
    durs = sorted(r["duration_min"] for r in rs)
    d["duration_p50"] = durs[len(durs) // 2] if durs else None
    return d


comp = {"funnel": desc(in_f), "non_funnel": desc(out_f)}
json.dump(comp, open(os.path.join(fbase, "..", "bias_%s.json" % FREEZE), "w"), indent=2)
print(json.dumps(comp, indent=1)[:2000])
