"""gate.py v1.0.0 — historical gate as code (spec v1.0.4, T18).

Loads the frozen spec YAML (sha-pinned), eval/oof.csv and eval/metrics.json,
applies ALL historical gate conditions per candidate model.
Candidates: linear, lgbm_A, lgbm_B, lgbm_C (baseline is reference only).
Prints per-model PASS/FAIL with failed conditions; overall conclusion:
any PASS -> FORWARD_ELIGIBLE list; none -> STOP (no forward).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

import metrics as M

GATE_VERSION = "v1.0.0"
SPEC_SHA256 = "2dce310ce3d4917061109d154520b0eb932dc9894b88b787d00fd32b0898071c"
CANDIDATES = ["linear", "lgbm_A", "lgbm_B", "lgbm_C"]
FOLDS = ["F1", "F2", "F3", "F4", "F5", "F6"]


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(base):
    path = os.path.join(base, "experiment_spec.yaml")
    assert sha_of(path) == SPEC_SHA256, "spec hash mismatch: %s" % sha_of(path)
    return yaml.safe_load(open(path))


def main(evaldir=None, base=None):
    b = base or os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("MKTREL_ROOT", os.path.dirname(b))
    evaldir = evaldir or os.path.join(root, "eval")
    spec = load_spec(b)
    g = spec["historical_gate_all_required"]
    met = json.load(open(os.path.join(evaldir, "metrics.json")))
    oof = pd.read_csv(os.path.join(evaldir, "oof.csv"), dtype={"market_id": str})

    overall = {}
    for m in ["baseline"] + CANDIDATES:
        sub = oof.dropna(subset=[m])
        y = sub["y"].to_numpy(dtype=np.float64)
        p = sub[m].to_numpy(dtype=np.float64)
        it, sl = M.cal_slope_intercept(y, p)
        overall[m] = {"logloss": M.logloss(y, p), "ece": M.ece(y, p),
                      "slope": sl, "intercept": it, "n": int(len(sub))}

    verdicts = {}
    for m in CANDIDATES:
        fails = []
        imp = overall["baseline"]["logloss"] - overall[m]["logloss"]
        if not imp >= g["logloss_improvement_overall_ge"]:
            fails.append("logloss_improvement_overall=%.5f" % imp)
        pos_folds = sum(1 for f in FOLDS
                        if met["folds"][f]["baseline"]["logloss"] - met["folds"][f][m]["logloss"] > 0)
        if not pos_folds >= g["logloss_improvement_positive_folds_ge"]:
            fails.append("logloss_pos_folds=%d" % pos_folds)
        e = met["econ"].get(m, {})
        if not e.get("canon_total", 0) and e.get("canon_total", 0) != 0:
            fails.append("no_econ")
        roi = e.get("roi", -9)
        if not roi > g["canonical_roi_overall_gt"]:
            fails.append("roi=%.5f" % roi)
        cpf = sum(1 for f in FOLDS if e.get("by_fold", {}).get(f, 0) > 0)
        if not cpf >= g["canonical_pnl_positive_folds_ge"]:
            fails.append("canon_pos_folds=%d" % cpf)
        if not e.get("n_signals", 0) >= g["signals_total_ge"]:
            fails.append("n_signals=%s" % e.get("n_signals"))
        sf = sum(1 for f in FOLDS if e.get("n_by_fold", {}).get(f, 0) >= 20)
        if not sf >= g["signals_ge_20_in_folds_ge"]:
            fails.append("folds_ge20=%d" % sf)
        fshare = e.get("fold_pos_share", [1.0, None])[0]
        if not fshare <= g["max_share_positive_pnl_one_fold_pct"] / 100.0:
            fails.append("fold_share=%.3f" % fshare)
        ashare = e.get("asset_pos_share", [1.0, None])[0]
        if not ashare <= g["max_share_positive_pnl_one_asset_pct"] / 100.0:
            fails.append("asset_share=%.3f" % ashare)
        if not overall[m]["ece"] <= g["ece_le"]:
            fails.append("ece=%.4f" % overall[m]["ece"])
        lo, hi = g["calibration_slope_within"]
        if not lo <= overall[m]["slope"] <= hi:
            fails.append("slope=%.3f" % overall[m]["slope"])
        if not e.get("maxdd_over_staked", -9) >= -g["maxdd_over_staked_le_pct"] / 100.0:
            fails.append("maxdd_staked=%.4f" % e.get("maxdd_over_staked"))
        ci_lo = e.get("ci95", [-9, 0])[0]
        if not ci_lo > g["ci95_lower_bound_mean_canon_per_1usd_stake_gt"]:
            fails.append("ci_lo=%.4f" % ci_lo)
        verdicts[m] = {"verdict": "PASS" if not fails else "FAIL",
                       "failed": fails,
                       "logloss_imp": round(imp, 5),
                       "n_signals": e.get("n_signals"),
                       "canon_total": round(e.get("canon_total", 0) or 0, 2)}
    eligible = [m for m in CANDIDATES if verdicts[m]["verdict"] == "PASS"]
    conclusion = "FORWARD_ELIGIBLE=%s" % eligible if eligible else "STOP_no_forward"
    out = {"gate_version": GATE_VERSION, "spec_sha256": SPEC_SHA256,
           "verdicts": verdicts, "conclusion": conclusion}
    with open(os.path.join(evaldir, "gate.json"), "w", newline="") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    for m in CANDIDATES:
        v = verdicts[m]
        print("%s %s imp=%s n=%s canon=%s fails=%s" % (
            m, v["verdict"], v["logloss_imp"], v["n_signals"],
            v["canon_total"], v["failed"]))
    print(conclusion)
    return out


if __name__ == "__main__":
    main()
