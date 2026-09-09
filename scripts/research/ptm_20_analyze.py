"""PTM steps 12-20: stats, weekly, bootstrap, sensitivity. Reads opportunity_ledger.csv."""
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else None
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID or "")
SEED = 20260911
MIN_DAYS = 10
BOOT = 2000


def load():
    rows = []
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("selection_status") != "OK":
                continue
            r["net_pnl"] = float(r["net_pnl"])
            r["gross_pnl"] = float(r["gross_pnl"])
            r["scenario_net_pnl"] = float(r["scenario_net_pnl"])
            r["target"] = int(r["target"])
            r["n_trades"] = 1
            rows.append(r)
    return rows


def agg(rs):
    import statistics
    n = len(rs)
    wins = [r for r in rs if r["target"] == 1]
    losses = [r for r in rs if r["target"] == 0]
    pnl = [r["net_pnl"] for r in rs]
    sp = [r["scenario_net_pnl"] for r in rs]
    days = sorted({r["calendar_date"] for r in rs})
    curve, peak, dd = 0.0, 0.0, 0.0
    for r in sorted(rs, key=lambda x: x["decision_at"]):
        curve += r["net_pnl"]
        peak = max(peak, curve, 0.0)
        dd = max(dd, peak - curve)
    top = sorted(pnl, reverse=True)
    top5 = sum(top[:5])
    avg_w = sum(r["net_pnl"] for r in wins) / len(wins) if wins else 0.0
    avg_l = sum(r["net_pnl"] for r in losses) / len(losses) if losses else 0.0
    return {
        "n": n, "days": len(days),
        "win_rate": round(len(wins) / n, 4),
        "gross_pnl": round(sum(r["gross_pnl"] for r in rs), 4),
        "net_pnl": round(sum(pnl), 4),
        "scenario_net_pnl": round(sum(sp), 4),
        "expectancy": round(sum(pnl) / n, 6),
        "avg_win": round(avg_w, 4), "avg_loss": round(avg_l, 4),
        "payoff": round(avg_w / abs(avg_l), 4) if avg_l else None,
        "max_drawdown": round(dd, 4),
        "median": round(statistics.median(pnl), 4),
        "top5_share": round(top5 / sum(pnl), 4) if sum(pnl) else None,
    }


def bootstrap_days(rows, seed=SEED, reps=BOOT):
    import random
    byday = defaultdict(list)
    for r in rows:
        byday[r["calendar_date"]].append(r)
    days = sorted(byday)
    if len(days) < MIN_DAYS:
        return {"status": "INSUFFICIENT_SAMPLE", "days": len(days)}
    rng = random.Random(seed)
    exps = []
    for _ in range(reps):
        sample = [rng.choice(days) for _ in days]
        p = sum(sum(r["net_pnl"] for r in byday[d]) for d in sample)
        c = sum(len(byday[d]) for d in sample)
        exps.append(p / c if c else 0.0)
    exps.sort()
    return {"status": "OK", "days": len(days),
            "exp_ci95": [round(exps[int(0.025 * reps)], 6), round(exps[int(0.975 * reps)], 6)]}


def main():
    rows = load()
    print("valid rows:", len(rows))
    # 12. main table
    groups = defaultdict(list)
    for r in rows:
        groups[(r["asset"], r["time_left_target"], r["entry_variant"], r["price_bin"])].append(r)
    table = []
    for key in sorted(groups):
        a = agg(groups[key])
        a.update({"asset": key[0], "entry_time": key[1], "variant": key[2], "price_bin": key[3]})
        a.update(bootstrap_days(groups[key]))
        table.append(a)
    with open(os.path.join(BASE, "price_time_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # 14. side comparison
    side = defaultdict(list)
    for r in rows:
        side[(r["entry_variant"], r["price_bin"])].append(r)
    with open(os.path.join(BASE, "side_comparison.csv"), "w", newline="") as f:
        recs = []
        for key in sorted(side):
            a = agg(side[key])
            a.update({"variant": key[0], "price_bin": key[1]})
            recs.append(a)
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    # 17-18. weekly
    weeks = defaultdict(list)
    for r in rows:
        dt = datetime.fromisoformat(r["calendar_date"])
        weeks[dt.strftime("%Y-W%V")].append(r)
    with open(os.path.join(BASE, "weekly_results.csv"), "w", newline="") as f:
        recs = []
        for wk in sorted(weeks):
            a = agg(weeks[wk])
            a.update({"week": wk})
            recs.append(a)
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    # 20. sensitivity
    sens = []
    for mp in (0.30, 0.35, 0.40, 0.50):
        for t in (12, 8, 5):
            rs = [r for r in rows if r["best_ask"] and float(r["best_ask"]) <= mp and int(r["time_left_target"]) == t]
            if not rs:
                continue
            a = agg(rs)
            b = bootstrap_days(rs)
            pnl_sorted = sorted((r["net_pnl"] for r in rs), reverse=True)
            tot = sum(r["net_pnl"] for r in rs)
            sens.append({"max_price": mp, "entry_time": t, **a,
                         "ci95": b.get("exp_ci95"), "status": b["status"], "days_n": b["days"],
                         "top1": round(pnl_sorted[0], 4),
                         "top3_share": round(sum(pnl_sorted[:3]) / tot, 4) if tot else None,
                         "top5_share": round(sum(pnl_sorted[:5]) / tot, 4) if tot else None,
                         "ex_top5_pnl": round(tot - sum(pnl_sorted[:5]), 4)})
    with open(os.path.join(BASE, "sensitivity_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)
    json.dump({"run": RUN_ID, "groups": len(table)}, open(os.path.join(BASE, "bootstrap_results.json"), "w"), indent=2)
    print("groups:", len(table), "sens rows:", len(sens))


main()
