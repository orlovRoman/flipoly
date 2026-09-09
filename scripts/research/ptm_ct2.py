"""PTM CT v2: classify from frozen history, join by opportunity_id (items 4,11,19,20).

Reads v2 ledger + freeze chunks (streamed). Writes ct_states.csv keyed by
opportunity_id + comparison_with_ct.csv grouped by policy/rule/asset/variant/bin.
"""
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
FREEZE = sys.argv[2] if len(sys.argv) > 2 else "freeze_funnel"


def main():
    sys.path.insert(0, os.path.abspath("scripts/research"))
    from polyflip.research.regime_features import classify_local_regime
    # targets: GRID YES-side rows + FUNNEL YES-side rows (parity with v1 CT scope)
    targets = defaultdict(list)
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("selection_status") == "OK" and r.get("side") == "YES"
                    and r.get("entry_policy") in ("GRID", "FUNNEL")):
                targets[r["market_id"]].append(
                    (r["opportunity_id"], datetime.fromisoformat(r["decision_at"]),
                     r["entry_variant"], r["asset"], r["price_bin"], float(r["gross_pnl"])))
    print("target rows:", sum(len(v) for v in targets.values()))
    states = {}
    fbase = os.path.join(BASE, "..", "_freeze", FREEZE)
    meta = json.load(open(os.path.join(fbase, "meta.json"), encoding="utf-8"))
    done = 0
    for f in meta["files"]:
        hist = defaultdict(list)
        with gzip.open(os.path.join(fbase, f["path"]), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["market_id"] in targets and r["mid_price"]:
                    hist[r["market_id"]].append(
                        (datetime.fromisoformat(r["recorded_at"]), float(r["mid_price"])))
        for mid, rows in hist.items():
            rows.sort()
            ts = [t for t, _ in rows]
            px = [p for _, p in rows]
            for oid, dec, variant, asset, pbin, pnl in targets.get(mid, []):
                window = [p for t, p in zip(ts, px) if dec - timedelta(minutes=15) <= t <= dec]
                if len(window) >= 3:
                    try:
                        states[oid] = classify_local_regime(window, min_observations=3).get("state", "UNCERTAIN")
                    except Exception:
                        states[oid] = "ERROR"
                else:
                    states[oid] = "INSUFFICIENT_HISTORY"
                done += 1
        print("ct2 file done, classified:", done, flush=True)
    with open(os.path.join(BASE, "ct_states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["opportunity_id", "market_id", "ct_state"])
        w.writeheader()
        for mid, rows in targets.items():
            for oid, dec, variant, asset, pbin, pnl in rows:
                w.writerow({"opportunity_id": oid, "market_id": mid,
                            "ct_state": states.get(oid, "MISSING")})
    groups = defaultdict(list)
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("selection_status") == "OK" and r.get("entry_policy") in ("GRID", "FUNNEL")
                    and r.get("entry_variant") == "YES_OUTSIDER" and r.get("opportunity_id") in states):
                groups[(r["entry_policy"], r["asset"], r["entry_rule"], r["price_bin"])].append(
                    (float(r["gross_pnl"]), states[r["opportunity_id"]]))
    comp = []
    for key in sorted(groups, key=str):
        rs = groups[key]
        base = sum(p for p, _ in rs)
        kept = [(p, s) for p, s in rs if s == "REVERSION"]
        kpnl = sum(p for p, s in kept)
        prevented = sum(-p for p, s in rs if s != "REVERSION" and p < 0)
        missed = sum(p for p, s in rs if s != "REVERSION" and p > 0)
        comp.append({"entry_policy": key[0], "asset": key[1], "entry_rule": key[2], "price_bin": key[3],
                     "n_c0": len(rs), "gross_c0": round(base, 4),
                     "n_ct": len(kept), "gross_ct": round(kpnl, 4),
                     "prevented_losses": round(prevented, 4), "missed_wins": round(missed, 4),
                     "net_diff": round(kpnl - base, 4)})
    with open(os.path.join(BASE, "comparison_with_ct.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)
    from collections import Counter
    print("ct states:", dict(Counter(states.values())))
    for pol in ("GRID", "FUNNEL"):
        pn = sum(c["n_ct"] for c in comp if c["entry_policy"] == pol)
        pg = sum(c["gross_ct"] for c in comp if c["entry_policy"] == pol)
        print("%s CT: n=%d gross=%.4f" % (pol, pn, pg))
    print("OUTDIR=" + BASE)


main()
