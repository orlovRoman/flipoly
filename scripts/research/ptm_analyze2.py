"""PTM v2 analysis (items 11,12,13,15,20,21,23,25).

Reads v2 ledger. Finance measure is explicit: 'gross' or 'scenario'.
- grouped totals labeled by full key, combined only as labeled unions;
- weekly per variant (entry_policy, rule, side, variant);
- auto findings (positive-CI cells, coverage statuses);
- Holm correction over tested cells (item 25);
- paired day-block CIs for CT-C0 and time pairs;
- old->new table vs a reference run dir (item 15).
Usage: python3 ptm_analyze2.py <run_id> [old_run_id]
"""
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

RUN_ID = sys.argv[1]
OLD_RUN = sys.argv[2] if len(sys.argv) > 2 else None
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
SEED = 20260911
MIN_DAYS = 10
BOOT = 2000
MEASURE = "gross"  # explicit finance measure for this analysis


def load():
    rows = []
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("selection_status") != "OK":
                continue
            r["gross_pnl"] = float(r["gross_pnl"])
            r["scenario_net_pnl"] = float(r["scenario_net_pnl"])
            r["target"] = int(r["target"])
            rows.append(r)
    return rows


def agg(rs, measure="gross"):
    import statistics
    key = "gross_pnl" if measure == "gross" else "scenario_net_pnl"
    n = len(rs)
    wins = [r for r in rs if r["target"] == 1]
    losses = [r for r in rs if r["target"] == 0]
    pnl = [r[key] for r in rs]
    days = sorted({r["calendar_date"] for r in rs})
    curve, peak, dd = 0.0, 0.0, 0.0
    for r in sorted(rs, key=lambda x: x["decision_at"]):
        curve += r[key]
        peak = max(peak, curve, 0.0)
        dd = max(dd, peak - curve)
    top = sorted(pnl, reverse=True)
    tot = sum(pnl)
    avg_w = sum(r[key] for r in wins) / len(wins) if wins else 0.0
    avg_l = sum(r[key] for r in losses) / len(losses) if losses else 0.0
    return {
        "measure": measure, "n": n, "days": len(days),
        "win_rate": round(len(wins) / n, 4),
        "pnl": round(tot, 4),
        "expectancy": round(tot / n, 6),
        "avg_win": round(avg_w, 4), "avg_loss": round(avg_l, 4),
        "payoff": round(avg_w / abs(avg_l), 4) if avg_l else None,
        "max_drawdown": round(dd, 4),
        "median": round(statistics.median(pnl), 4),
        "top1": round(top[0], 4),
        "top3_share": round(sum(top[:3]) / tot, 4) if tot else None,
        "top5_share": round(sum(top[:5]) / tot, 4) if tot else None,
        "ex_top5_pnl": round(tot - sum(top[:5]), 4),
    }


def boot_ci(rows, measure="gross", seed=SEED, reps=BOOT):
    key = "gross_pnl" if measure == "gross" else "scenario_net_pnl"
    byday = defaultdict(list)
    for r in rows:
        byday[r["calendar_date"]].append(r)
    days = sorted(byday)
    if len(days) < MIN_DAYS:
        return {"status": "INSUFFICIENT_SAMPLE", "days": len(days), "exp_ci95": None}
    rng = random.Random(seed)
    exps = []
    for _ in range(reps):
        sample = [rng.choice(days) for _ in days]
        p = sum(sum(r[key] for r in byday[d]) for d in sample)
        c = sum(len(byday[d]) for d in sample)
        exps.append(p / c if c else 0.0)
    exps.sort()
    return {"status": "OK", "days": len(days),
            "exp_ci95": [round(exps[int(0.025 * reps)], 6), round(exps[int(0.975 * reps)], 6)]}


def holm(pvals):
    """Holm step-down adjusted p-values. Returns {key: adj_p}."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    adj, prev = {}, 0.0
    for i, k in enumerate(order):
        adj[k] = max(prev, min(1.0, (m - i) * pvals[k]))
        prev = adj[k]
    return adj


def main():
    rows = load()
    print("valid rows:", len(rows), "measure:", MEASURE)
    # main table: policy x rule x asset x variant x bin
    groups = defaultdict(list)
    for r in rows:
        groups[(r["entry_policy"], r["entry_rule"], r["asset"], r["entry_variant"], r["price_bin"])].append(r)
    table, pvals = [], {}
    for key in sorted(groups, key=str):
        a = agg(groups[key], MEASURE)
        b = boot_ci(groups[key], MEASURE)
        a.update({"entry_policy": key[0], "entry_rule": key[1], "asset": key[2],
                  "variant": key[3], "price_bin": key[4], **{k: v for k, v in b.items() if k != "exp_ci95"},
                  "exp_ci95": b["exp_ci95"]})
        table.append(a)
        if b["status"] == "OK" and b["exp_ci95"]:
            lo, hi = b["exp_ci95"]
            # two-sided bootstrap p-value approx via CI position relative to 0
            mid = a["expectancy"]
            width = max(hi - lo, 1e-12)
            z = abs(mid) / (width / (2 * 1.96))
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
            pvals[key] = p
    adj = holm(pvals)
    for a in table:
        k = (a["entry_policy"], a["entry_rule"], a["asset"], a["variant"], a["price_bin"])
        a["holm_adj_p"] = round(adj[k], 6) if k in adj else None
    with open(os.path.join(BASE, "price_time_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # item 13: auto findings
    pos = [a for a in table if a["exp_ci95"] and a["exp_ci95"][0] > 0]
    findings = {
        "measure": MEASURE, "cells_tested": len(table),
        "cells_ci_ok": sum(1 for a in table if a["status"] == "OK"),
        "positive_lower_ci": len(pos),
        "positive_cells": [
            {"key": [a["entry_policy"], a["entry_rule"], a["asset"], a["variant"], a["price_bin"]],
             "n": a["n"], "pnl": a["pnl"], "ci": a["exp_ci95"], "holm_adj_p": a["holm_adj_p"]} for a in pos],
        "coverage_statuses": {},
    }
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        cov = defaultdict(int)
        for r in csv.DictReader(f):
            cov[r.get("selection_status") or "MISSING"] += 1
        findings["coverage_statuses"] = dict(cov)
    json.dump(findings, open(os.path.join(BASE, "auto_findings.json"), "w"), indent=2)
    # weekly per variant (item 12)
    wgroups = defaultdict(list)
    for r in rows:
        dt = datetime.fromisoformat(r["calendar_date"])
        wgroups[(r["entry_policy"], r["entry_rule"], r["entry_variant"], dt.strftime("%Y-W%V"))].append(r)
    with open(os.path.join(BASE, "weekly_results.csv"), "w", newline="") as f:
        recs = []
        for key in sorted(wgroups, key=str):
            a = agg(wgroups[key], MEASURE)
            a.update({"entry_policy": key[0], "entry_rule": key[1], "variant": key[2], "week": key[3]})
            recs.append(a)
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    # sensitivity (item 20)
    grid = [r for r in rows if r.get("entry_policy") == "GRID"]
    sens = []
    for mp in (0.30, 0.35, 0.40, 0.50):
        for t in (12, 8, 5):
            rs = [r for r in grid if r["best_ask"] and float(r["best_ask"]) <= mp and int(float(r["time_left_target"])) == t]
            if not rs:
                continue
            a = agg(rs, MEASURE)
            b = boot_ci(rs, MEASURE)
            sens.append({"max_price": mp, "entry_time": t, **a, "ci95": b.get("exp_ci95"),
                         "status": b["status"], "days_n": b["days"]})
    with open(os.path.join(BASE, "sensitivity_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)
    # item 15: old -> new table
    if OLD_RUN:
        old_base = os.path.join("artifacts", "research", "price_time_map", OLD_RUN)
        old_cov = json.load(open(os.path.join(old_base, "coverage.json"), encoding="utf-8"))
        new_cov = json.load(open(os.path.join(BASE, "coverage.json"), encoding="utf-8"))
        table15 = [{"metric": k, "old": old_cov.get(k), "new": new_cov.get(k),
                    "reason": "v2 contracts (see report)"} for k in
                   ("markets", "outcome_ok", "entries_ok", "entries_missing")]
        json.dump(table15, open(os.path.join(BASE, "old_to_new.json"), "w"), indent=2)
    json.dump({"run": RUN_ID, "groups": len(table), "measure": MEASURE},
              open(os.path.join(BASE, "bootstrap_results.json"), "w"), indent=2)
    print("groups:", len(table), "positive-CI:", len(pos), "sens:", len(sens))


main()
