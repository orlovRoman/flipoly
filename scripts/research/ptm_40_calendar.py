"""PTM step 17: calendar splits (asset x hour-group x weekday). Reads ledger."""
import csv
import os
import sys
from collections import defaultdict

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)


def hgroup(h):
    h = int(h)
    if h < 6:
        return "00-06"
    if h < 12:
        return "06-12"
    if h < 18:
        return "12-18"
    return "18-24"


rows = []
with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
    for r in csv.DictReader(f):
        if r.get("selection_status") != "OK":
            continue
        r["net_pnl"] = float(r["net_pnl"])
        rows.append(r)

groups = defaultdict(list)
for r in rows:
    groups[(r["asset"], hgroup(r["hour_utc"]), r["weekday"], r.get("entry_policy") or "GRID", r["entry_variant"])].append(r)

recs = []
for key in sorted(groups, key=str):
    rs = groups[key]
    n = len(rs)
    pnl = sum(r["net_pnl"] for r in rs)
    days = len({r["calendar_date"] for r in rs})
    wins = sum(1 for r in rs if r["target"] == "1")
    recs.append({"asset": key[0], "hour_group_utc": key[1], "weekday": key[2],
                 "entry_policy": key[3], "variant": key[4], "n": n, "days": days,
                 "win_rate": round(wins / n, 4), "net_pnl": round(pnl, 4),
                 "expectancy": round(pnl / n, 6)})
with open(os.path.join(BASE, "asset_time_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
    w.writeheader()
    w.writerows(recs)
print("calendar groups:", len(recs))
