"""Stage 1/2 regression lock for v2 runs (read-only). Recomputes headline
numbers directly from ledger + ct_states.csv. Fails loudly on mismatch.

Usage: python3 ptm_verify2.py <run_id>
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
fails = []


def check(name, got, exp, tol=1e-6):
    ok = abs(got - exp) <= tol if isinstance(exp, float) else got == exp
    print(("PASS" if ok else "FAIL"), name, "got=%r exp=%r" % (got, exp))
    if not ok:
        fails.append(name)


ct = {}
with open(os.path.join(BASE, "ct_states.csv")) as f:
    for r in csv.DictReader(f):
        ct[r["opportunity_id"]] = r["ct_state"]

ok_rows, pend, fee_e, net_e, oids = 0, 0, 0, 0, []
stat = Counter()
led = list(csv.DictReader(open(os.path.join(BASE, "opportunity_ledger.csv"))))
for row in led:
    stat[row["selection_status"]] += 1
    if row["selection_status"] != "OK":
        continue
    ok_rows += 1
    oids.append(row["opportunity_id"])
    if row["final_outcome"] == "PENDING":
        pend += 1
    if not (row.get("fee") or "").strip():
        fee_e += 1
    if not (row.get("net_pnl") or "").strip():
        net_e += 1
check("ok_rows", ok_rows, 35752)
check("pending_in_ok", pend, 0)
check("fee_empty", fee_e, ok_rows)
check("net_empty", net_e, ok_rows)
check("dup_oid", len(oids) - len(set(oids)), 0)
check("empty_price_bin_ok", sum(1 for r in led if r["selection_status"] == "OK" and not (r.get("price_bin") or "").strip()), 0)


def ct_scope(policy, variant="YES_OUTSIDER"):
    n, g = 0, 0.0
    for r in led:
        if (r["selection_status"] == "OK" and r.get("entry_policy") == policy
                and r.get("entry_variant") == variant and ct.get(r["opportunity_id"]) == "REVERSION"):
            n += 1
            g += float(r["gross_pnl"])
    return n, round(g, 4)


n, g = ct_scope("GRID")
check("grid_ct_n", n, 2978)
check("grid_ct_gross", g, -27.6319, tol=0.01)
n, g = ct_scope("FUNNEL")
check("funnel_ct_n", n, 563)
check("funnel_ct_gross", g, -20.7471, tol=0.01)

t5 = [float(r["gross_pnl"]) for r in led
      if r["selection_status"] == "OK" and r.get("entry_policy") == "GRID"
      and r.get("entry_rule") == "T-5" and r.get("entry_variant") == "YES_OUTSIDER"
      and ct.get(r["opportunity_id"]) == "REVERSION"]
t5.sort(reverse=True)
check("t5_n", len(t5), 915)
check("t5_gross", round(sum(t5), 2), 100.10, tol=0.01)
check("t5_top5", round(sum(t5[:5]), 2), 195.00, tol=0.01)
check("t5_ex_top5", round(sum(t5) - sum(t5[:5]), 2), -94.90, tol=0.01)

byday = defaultdict(float)
for r in led:
    if (r["selection_status"] == "OK" and r.get("entry_policy") == "GRID"
            and r.get("entry_rule") == "T-5" and r.get("entry_variant") == "YES_OUTSIDER"
            and ct.get(r["opportunity_id"]) == "REVERSION"):
        byday[r["calendar_date"]] += float(r["gross_pnl"])
big = max(byday.items(), key=lambda x: x[1])
check("t5_biggest_day", big[0], "2026-08-28")
check("t5_biggest_day_pnl", round(big[1], 2), 92.49, tol=0.01)

af = json.load(open(os.path.join(BASE, "auto_findings.json")))
# P1-1 fix: significance = day-block recentered bootstrap null + Holm; no normal approx.
check("perm_method", "NO normal approximation" in af.get("method", ""), True)
check("positive_cells", len(af["positive_cells"]), 15)
surv = [p for p in af["positive_cells"]
        if p.get("holm_adj_p") is not None and p["holm_adj_p"] < 0.05]
check("holm_survivors_at_0_05", len(surv), 0)
best = min(p["holm_adj_p"] for p in af["positive_cells"] if p.get("holm_adj_p") is not None)
check("holm_best_adj", best, 0.1545, tol=0.001)

u = json.load(open(os.path.join(BASE, "..", "_freeze", "universe_freeze_independent.json")))
check("universe_markets", len(u["markets"]), 14820)  # 12745 funnel (incl 1 PENDING) + 2075 non-funnel
pend_mids = {m["market_id"] for m in u["markets"] if m["outcome"] == "PENDING"}
check("universe_pending", pend_mids, {"3337747"})
check("pending_excluded_from_ok",
      sum(1 for r in led if r["selection_status"] == "OK" and r["market_id"] in pend_mids), 0)
b = json.load(open(os.path.join(BASE, "..", "_freeze", "bias_freeze_independent.json")))
check("bias_funnel", b["funnel"]["n"], 12745)
check("bias_nonfunnel", b["non_funnel"]["n"], 2075)

print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
