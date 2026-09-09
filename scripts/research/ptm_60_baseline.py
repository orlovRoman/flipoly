"""Stage 1 item 1+13+23: lock seeded baseline numbers as regression checks (read-only)."""
import csv
import os
import sys
from collections import defaultdict

BASE = os.path.join("artifacts", "research", "price_time_map", "ptm_20260909_082348")

ct = {}
with open(os.path.join(BASE, "ct_states.csv"), newline="") as f:
    for r in csv.DictReader(f):
        ct[int(r["lid"])] = r["ct_state"]

grid_n = grid_pnl = fun_n = fun_pnl = 0
grid_t5_n = grid_t5_pnl = 0
by_asset_t5 = defaultdict(lambda: [0, 0.0])
wins_t5 = []
with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if r.get("selection_status") != "OK" or r.get("entry_variant") != "YES_OUTSIDER":
            continue
        if ct.get(i) != "REVERSION":
            continue
        pnl = float(r["net_pnl"])
        if r.get("entry_policy") == "GRID":
            grid_n += 1
            grid_pnl += pnl
            if r.get("time_left_target") == "5":
                grid_t5_n += 1
                grid_t5_pnl += pnl
                by_asset_t5[r["asset"]][0] += 1
                by_asset_t5[r["asset"]][1] += pnl
                wins_t5.append(pnl)
        elif r.get("entry_policy") == "FUNNEL":
            fun_n += 1
            fun_pnl += pnl

print("GRID CT: n=%d gross=%.6f" % (grid_n, grid_pnl))
print("FUNNEL CT: n=%d gross=%.6f" % (fun_n, fun_pnl))
print("COMBINED: n=%d gross=%.6f" % (grid_n + fun_n, grid_pnl + fun_pnl))
print("GRID CT T-5: n=%d gross=%.6f" % (grid_t5_n, grid_t5_pnl))
print("T-5 by asset:", {k: (v[0], round(v[1], 2)) for k, v in sorted(by_asset_t5.items())})
wins_t5.sort(reverse=True)
top5 = sum(wins_t5[:5])
print("T-5 top5=%.2f ex-top5=%.6f" % (top5, grid_t5_pnl - top5))

# item 13: positive lower bounds in price_time_table
pos = total = 0
empty_pos = 0
with open(os.path.join(BASE, "price_time_table.csv"), newline="") as f:
    for r in csv.DictReader(f):
        total += 1
        ci = (r.get("exp_ci95") or "").strip("[]")
        if not ci:
            continue
        try:
            lo = float(ci.split(",")[0])
        except ValueError:
            continue
        if lo > 0:
            pos += 1
            if not (r.get("price_bin") or "").strip():
                empty_pos += 1
print("positive-CI cells: %d/%d (empty-bin: %d)" % (pos, total, empty_pos))

# empty price_bin rows in ledger
with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
    empty = sum(1 for r in csv.DictReader(f) if r.get("selection_status") == "OK" and not (r.get("price_bin") or "").strip())
print("empty price_bin OK rows:", empty)
