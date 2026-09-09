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
        return {"status": "INSUFFICIENT_SAMPLE", "days": len(days), "exp_ci95": None}
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
    def tlabel(r):
        return ("GRID", int(float(r["time_left_target"]))) if r.get("entry_policy") == "GRID" else ("FUNNEL", -1)
    # 12. main table
    groups = defaultdict(list)
    for r in rows:
        groups[(r["asset"], tlabel(r), r["entry_variant"], r["price_bin"])].append(r)
    table = []
    for key in sorted(groups, key=str):
        a = agg(groups[key])
        a.update({"asset": key[0], "entry_policy": key[1][0], "entry_time": key[1][1] if key[1][0] == "GRID" else "FUNNEL",
                  "variant": key[2], "price_bin": key[3]})
        a.update(bootstrap_days(groups[key]))
        table.append(a)
    with open(os.path.join(BASE, "price_time_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # 14. side comparison (per policy)
    side = defaultdict(list)
    for r in rows:
        side[(r.get("entry_policy") or "GRID", r["entry_variant"], r["price_bin"])].append(r)
    with open(os.path.join(BASE, "side_comparison.csv"), "w", newline="") as f:
        recs = []
        for key in sorted(side, key=str):
            a = agg(side[key])
            a.update({"entry_policy": key[0], "variant": key[1], "price_bin": key[2]})
            a.update(bootstrap_days(side[key]))
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
    grid = [r for r in rows if r.get("entry_policy") == "GRID"]
    for mp in (0.30, 0.35, 0.40, 0.50):
        for t in (12, 8, 5):
            rs = [r for r in grid if r["best_ask"] and float(r["best_ask"]) <= mp and int(float(r["time_left_target"])) == t]
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
    # 13/19. paired entry-time diffs on same markets, day-clustered bootstrap
    import random as _random
    grid_by_mkt = defaultdict(lambda: defaultdict(list))
    for r in grid:
        grid_by_mkt[r["market_id"]][int(float(r["time_left_target"]))].append(r)
    pairs, paired_rows = [], []
    for (ta, tb) in ((12, 8), (12, 5), (8, 5)):
        per_market = {}
        for m, d in grid_by_mkt.items():
            if ta in d and tb in d:
                per_market[m] = (sum(r["net_pnl"] for r in d[ta]), sum(r["net_pnl"] for r in d[tb]))
        if not per_market:
            continue
        day_of = {}
        for m in per_market:
            day_of[m] = next(r["calendar_date"] for r in grid if r["market_id"] == m and int(float(r["time_left_target"])) == ta)
        byday = defaultdict(list)
        for m, (pa_, pb_) in per_market.items():
            byday[day_of[m]].append(pb_ - pa_)
        days = sorted(byday)
        diffs = [pb - pa for pa, pb in per_market.values()]
        rec = {"pair": f"T-{ta}_vs_T-{tb}", "markets": len(per_market), "days": len(days),
               "mean_diff": round(sum(diffs) / len(diffs), 6)}
        if len(days) >= MIN_DAYS:
            rng = _random.Random(SEED)
            means = []
            for _ in range(BOOT):
                s = [rng.choice(days) for _ in days]
                vals = [v for d in s for v in byday[d]]
                means.append(sum(vals) / len(vals))
            means.sort()
            rec.update({"status": "OK", "ci95": [round(means[int(0.025 * BOOT)], 6), round(means[int(0.975 * BOOT)], 6)]})
        else:
            rec.update({"status": "INSUFFICIENT_SAMPLE", "ci95": None})
        pairs.append(rec)
        for m, (pa_, pb_) in per_market.items():
            paired_rows.append({"market_id": m, "pair": rec["pair"], "pnl_a": round(pa_, 4), "pnl_b": round(pb_, 4),
                               "diff": round(pb_ - pa_, 4), "calendar_date": day_of[m]})
    with open(os.path.join(BASE, "paired_time_diffs.csv"), "w", newline="") as f:
        if pairs:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
    json.dump({"run": RUN_ID, "groups": len(table), "paired": pairs},
              open(os.path.join(BASE, "bootstrap_results.json"), "w"), indent=2)
    print("groups:", len(table), "sens rows:", len(sens), "pairs:", len(pairs))


main()
