"""Print overall C0 / CT / FUNNEL totals for report.md (read-only)."""
import csv
import os
import sys
from collections import defaultdict

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
tot = defaultdict(lambda: [0, 0.0, 0.0])
with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
    for r in csv.DictReader(f):
        if r.get("selection_status") != "OK":
            continue
        k = (r.get("entry_policy"), r["entry_variant"])
        tot[k][0] += 1
        tot[k][1] += float(r["net_pnl"])
        tot[k][2] += float(r["scenario_net_pnl"])
for k in sorted(tot, key=str):
    n, net, snet = tot[k]
    print(k, "n=%d net=%.2f scenario=%.2f exp=%.4f" % (n, net, snet, net / n))
ct = {}
with open(os.path.join(BASE, "ct_states.csv"), newline="") as f:
    for r in csv.DictReader(f):
        ct[int(r["lid"])] = r["ct_state"]
from collections import Counter
print("ct:", dict(Counter(ct.values())))
# CT-filtered YES_OUTSIDER gross total
sel = [0.0, 0]
with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if r.get("selection_status") == "OK" and r.get("entry_variant") == "YES_OUTSIDER" and ct.get(i) == "REVERSION":
            sel[0] += float(r["net_pnl"])
            sel[1] += 1
print("CT-kept YES_OUTSIDER: n=%d net=%.2f" % (sel[1], sel[0]))
