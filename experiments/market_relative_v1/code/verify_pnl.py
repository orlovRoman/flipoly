"""verify_pnl.py — independent PnL recomputation (T17 self-check).

Pure stdlib (csv/math): re-aggregates signals.csv per model, verifies the
per-row identity canon == executable - fee07 - slip_extra (no double count),
and compares totals/means/maxDD against eval/metrics.json within 1e-6.
Exit 0 on match, 1 with diff report otherwise.
"""
import csv
import json
import math
import os
import sys

VERIFY_VERSION = "v1.0.0"


def main(evaldir=None):
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("MKTREL_ROOT", os.path.dirname(base))
    evaldir = evaldir or os.path.join(root, "eval")
    met = json.load(open(os.path.join(evaldir, "metrics.json")))
    per_model = {}
    bad_rows = 0
    checked = 0
    with open(os.path.join(evaldir, "signals.csv"), newline="") as f:
        for r in csv.DictReader(f):
            m = r["model"]
            raw = float(r["raw"])
            exe = float(r["executable"])
            fee = float(r["fee07"])
            se = float(r["slip_extra"])
            cn = float(r["canon_net"])
            if abs(cn - (exe - fee - se)) > 1e-6:
                bad_rows += 1
            checked += 1
            d = per_model.setdefault(m, {"n": 0, "raw": 0.0, "canon": 0.0,
                                         "eq": 0.0, "peak": 0.0, "dd": 0.0,
                                         "rows": []})
            d["n"] += 1
            d["raw"] += raw
            d["canon"] += cn
            d["rows"].append((r["end3"], r["market_id"], cn))
    print("rows_checked=%d identity_violations=%d" % (checked, bad_rows))
    ok = bad_rows == 0
    for m, d in sorted(per_model.items()):
        e = met["econ"][m]
        d["rows"].sort()
        eq = 0.0
        peak = 0.0
        dd = 0.0
        for _, _, cn in d["rows"]:
            eq += cn
            peak = max(peak, eq)
            dd = min(dd, eq - peak)
        match = (d["n"] == e["n_signals"]
                 and abs(d["raw"] - e["raw_total"]) < 1e-6
                 and abs(d["canon"] - e["canon_total"]) < 1e-6
                 and abs(dd - e["maxdd"]) < 1e-6)
        print("%s n=%d raw=%+.4f canon=%+.4f dd=%+.4f match=%s" % (
            m, d["n"], d["raw"], d["canon"], dd, match))
        ok = ok and match
    print("VERIFY " + ("OK" if ok else "DIFF"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
