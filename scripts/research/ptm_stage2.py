"""PTM stage 2 items 19-23: C0/CT time×asset table, paired CIs, concentration.

Reads v2 ledger + ct_states.csv (opportunity_id key).
Writes ct_time_asset.csv, ct_paired.csv, ct_t5_concentration.json.
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
SEED = 20260911
BOOT = 2000
MIN_DAYS = 10


def main():
    ct = {}
    with open(os.path.join(BASE, "ct_states.csv"), newline="") as f:
        for r in csv.DictReader(f):
            ct[r["opportunity_id"]] = r["ct_state"]
    rows = []
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("selection_status") == "OK" and r.get("entry_policy") == "GRID"
                    and r.get("entry_variant") == "YES_OUTSIDER" and r.get("opportunity_id") in ct):
                r["gross_pnl"] = float(r["gross_pnl"])
                r["ct"] = ct[r["opportunity_id"]]
                rows.append(r)
    # 19. time x asset table
    groups = defaultdict(list)
    for r in rows:
        groups[(r["asset"], r["entry_rule"])].append(r)
    table = []
    for key in sorted(groups, key=str):
        rs = groups[key]
        kept = [r for r in rs if r["ct"] == "REVERSION"]
        table.append({
            "asset": key[0], "entry_rule": key[1],
            "c0_n": len(rs), "c0_gross": round(sum(r["gross_pnl"] for r in rs), 4),
            "ct_n": len(kept), "ct_gross": round(sum(r["gross_pnl"] for r in kept), 4),
            "ct_coverage": round(len(kept) / len(rs), 4),
            "ct_missing_reason_top": "non-REVERSION",
            "days": len({r["calendar_date"] for r in rs}),
            "markets": len({r["market_id"] for r in rs}),
        })
    with open(os.path.join(BASE, "ct_time_asset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # 21. paired day-block CIs: CT expectancy and CT-C0 diff
    rng = random.Random(SEED)
    paired = []
    for key in sorted(groups, key=str):
        rs = groups[key]
        byday = defaultdict(list)
        for r in rs:
            byday[r["calendar_date"]].append(r)
        days = sorted(byday)
        rec = {"asset": key[0], "entry_rule": key[1], "days": len(days)}
        if len(days) >= MIN_DAYS:
            ct_exps, df_exps = [], []
            for _ in range(BOOT):
                s = [rng.choice(days) for _ in days]
                allr = [r for d in s for r in byday[d]]
                kept = [r for r in allr if r["ct"] == "REVERSION"]
                ct_exps.append(sum(r["gross_pnl"] for r in kept) / len(kept) if kept else 0.0)
                df_exps.append((sum(r["gross_pnl"] for r in kept) - sum(r["gross_pnl"] for r in allr)) / len(allr))
            ct_exps.sort()
            df_exps.sort()
            rec.update({"status": "OK",
                        "ct_exp_ci95": [round(ct_exps[int(0.025 * BOOT)], 6), round(ct_exps[int(0.975 * BOOT)], 6)],
                        "ct_minus_c0_ci95": [round(df_exps[int(0.025 * BOOT)], 6), round(df_exps[int(0.975 * BOOT)], 6)]})
        else:
            rec.update({"status": "INSUFFICIENT_SAMPLE", "ct_exp_ci95": None, "ct_minus_c0_ci95": None})
        paired.append(rec)
    with open(os.path.join(BASE, "ct_paired.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)
    # 22. CT time-pair comparison on common markets (CT-skip = zero position)
    trows = defaultdict(lambda: defaultdict(list))
    for r in rows:
        trows[r["market_id"]][r["entry_rule"]].append(r)
    pairs = []
    for (ra, rb) in (("T-12", "T-8"), ("T-12", "T-5"), ("T-8", "T-5")):
        common = [m for m in trows if ra in trows[m] and rb in trows[m]]
        da = [sum(r["gross_pnl"] for r in trows[m][ra] if r["ct"] == "REVERSION") for m in common]
        db = [sum(r["gross_pnl"] for r in trows[m][rb] if r["ct"] == "REVERSION") for m in common]
        diffs = [b - a for a, b in zip(da, db)]
        pairs.append({"pair": "%s_vs_%s_CT" % (ra, rb), "markets": len(common),
                      "mean_diff": round(sum(diffs) / len(diffs), 6) if diffs else None})
    # 23. T-5 CT concentration
    t5 = [r for r in rows if r["entry_rule"] == "T-5" and r["ct"] == "REVERSION"]
    pnls = sorted((r["gross_pnl"] for r in t5), reverse=True)
    tot = sum(pnls)
    bya = defaultdict(lambda: [0, 0.0])
    byw = defaultdict(lambda: [0, 0.0])
    byday = defaultdict(float)
    for r in t5:
        bya[r["asset"]][0] += 1
        bya[r["asset"]][1] += r["gross_pnl"]
        wk = datetime.fromisoformat(r["calendar_date"]).strftime("%Y-W%V")
        byw[wk][0] += 1
        byw[wk][1] += r["gross_pnl"]
        byday[r["calendar_date"]] += r["gross_pnl"]
    conc = {"n": len(t5), "gross": round(tot, 4),
            "by_asset": {k: [v[0], round(v[1], 2)] for k, v in sorted(bya.items())},
            "by_week": {k: [v[0], round(v[1], 2)] for k, v in sorted(byw.items())},
            "top1": round(pnls[0], 2), "top3_share": round(sum(pnls[:3]) / tot, 4) if tot else None,
            "top5": round(sum(pnls[:5]), 2), "ex_top5": round(tot - sum(pnls[:5]), 4),
            "biggest_day": max(byday.items(), key=lambda x: x[1]) if byday else None}
    # data completeness: winners vs losers quote/outcome presence (already 100% by construction; verify)
    conc["winner_rows_complete"] = sum(1 for r in t5 if r["gross_pnl"] > 0 and r["best_ask"] and r["final_outcome"])
    conc["loser_rows_complete"] = sum(1 for r in t5 if r["gross_pnl"] <= 0 and r["best_ask"] and r["final_outcome"])
    json.dump({"pairs": pairs, "t5_concentration": conc},
              open(os.path.join(BASE, "ct_stage2.json"), "w"), indent=2)
    print("ct pairs:", pairs)
    print("t5 conc:", json.dumps(conc)[:600])


main()
