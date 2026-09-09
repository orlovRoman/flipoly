"""Verify v2 ledger invariants (read-only)."""
import csv
import os
import sys
from collections import Counter

BASE = os.path.join("artifacts", "research", "price_time_map", sys.argv[1])
oids, bins, roles, delays = set(), Counter(), Counter(), []
dup = 0
net_nonempty = fee_nonempty = 0
n_ok = 0
with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
    for r in csv.DictReader(f):
        if r.get("selection_status") == "OK":
            n_ok += 1
            if r["opportunity_id"] in oids:
                dup += 1
            oids.add(r["opportunity_id"])
            bins[r.get("bin_status") or "MISSING"] += 1
            roles[r.get("role_basis") or "MISSING"] += 1
            if r.get("entry_policy") == "GRID":
                delays.append(float(r["entry_delay_sec"]))
            if (r.get("net_pnl") or "") != "":
                net_nonempty += 1
            if (r.get("fee") or "") != "":
                fee_nonempty += 1
print("ok rows:", n_ok, "dup oids:", dup)
print("net nonempty:", net_nonempty, "fee nonempty:", fee_nonempty)
print("bins:", dict(bins))
print("roles:", dict(roles))
if delays:
    print("delay max: %.1f n=%d over30: %d" % (max(delays), len(delays), sum(1 for d in delays if d > 30)))
