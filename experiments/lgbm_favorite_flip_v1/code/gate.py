"""gate.py — development gate verdict from frozen artifacts (spec v1.1.0).

Reads METRICS.json + ECON_SUMMARY.json. Evaluates 11 conditions per model
(L0..L3), priority L0->L1->L2->L3, max one candidate. Writes GATE.json.
Exit 0 always (verdict encoded); STOP unless all mandatory conditions hold.
"""
import json
import os

RUNDIR = r"D:\lgbm-favorite-flip-v1\runs"
MODELS = ["L0", "L1", "L2", "L3"]
POL = {"L0": "P2_L0", "L1": "P3_L1", "L2": "P4_L2", "L3": "P5_L3"}


def main():
    M = json.load(open(os.path.join(RUNDIR, "METRICS.json")))
    E = json.load(open(os.path.join(RUNDIR, "ECON_SUMMARY.json")))
    res = {}
    for m in MODELS:
        mc = M["models_cal"][m]
        c = {}
        pm, pl = M["paired"]["%s_vs_market" % m], M["paired"]["%s_vs_logreg" % m]
        c["logloss_better_than_market"] = bool(pm["by_market"]["lo"] > 0)
        c["logloss_better_than_logreg"] = bool(pl["by_market"]["lo"] > 0)
        wm = sum(1 for f in ["F1", "F2", "F3", "F4", "F5", "F6"]
                 if M["per_fold"][m][f]["logloss"]
                 < M["per_fold"]["market"][f])
        wl = sum(1 for f in ["F1", "F2", "F3", "F4", "F5", "F6"]
                 if M["per_fold"]["logreg"][f] is not None
                 and M["per_fold"][m][f]["logloss"]
                 < M["per_fold"]["logreg"][f])
        c["folds_won_vs_market"] = wm
        c["folds_won_vs_logreg"] = wl
        c["wins_at_least_4_of_6_folds"] = bool(wm >= 4 and wl >= 4)
        c["ci_logloss_improvement_above_zero"] = bool(
            pm["by_market"]["lo"] > 0 and pl["by_market"]["lo"] > 0)
        c["brier_not_worse"] = bool(
            mc["brier"] <= min(M["controls"]["market"]["brier"],
                               M["controls"]["logreg"]["brier"]) + 1e-6)
        c["calibration_acceptable"] = bool(
            0.7 <= mc["slope"] <= 1.3 and abs(mc["intercept"]) <= 0.3
            and mc["ece"] <= 0.05)
        p = E[POL[m]]
        c["canonical_net_above_both"] = bool(
            p["net"] > E["P0_market"]["net"]
            and p["net"] > E["P1_logreg"]["net"])
        nm = E["paired_net"]["net_%s_minus_net_P0_market" % POL[m]]["by_market"]
        nl = E["paired_net"]["net_%s_minus_net_P1_logreg" % POL[m]]["by_market"]
        c["ci_economic_increment_above_zero"] = bool(nm["lo"] > 0
                                                    and nl["lo"] > 0)
        c["not_concentrated"] = bool(
            p["net"] > 0 and (p["top1_day_share"] or 1) <= 0.5
            and (p["top1_mkt_share"] or 1) <= 0.25)
        c["max_drawdown_within_limit"] = bool(p["max_drawdown"] <= 5.0)
        cov_ok = (pm["n"] >= 1000 and pm["markets"] >= 500
                  and pm["flips"] >= 50 and pl["n"] >= 1000
                  and pl["markets"] >= 500 and pl["flips"] >= 50)
        c["coverage_sufficient"] = bool(cov_ok)
        c["pass_all"] = all(c[k] for k in [
            "logloss_better_than_market", "logloss_better_than_logreg",
            "wins_at_least_4_of_6_folds",
            "ci_logloss_improvement_above_zero", "brier_not_worse",
            "calibration_acceptable", "canonical_net_above_both",
            "ci_economic_increment_above_zero", "not_concentrated",
            "max_drawdown_within_limit", "coverage_sufficient"])
        res[m] = c
    cand = next((m for m in MODELS if res[m]["pass_all"]), None)
    if cand:
        verdict = "PASS_PAPER_SHADOW"
    else:
        any_mkt = any(res[m]["logloss_better_than_market"]
                      and res[m]["ci_logloss_improvement_above_zero"]
                      for m in MODELS)
        if not any_mkt:
            verdict = "STOP_NOT_BETTER_THAN_MARKET"
        else:
            any_lr = any(res[m]["logloss_better_than_logreg"]
                         and res[m]["ci_logloss_improvement_above_zero"]
                         for m in MODELS)
            if not any_lr:
                verdict = "STOP_NOT_BETTER_THAN_LOGREG"
            else:
                any_eco = any(res[m]["canonical_net_above_both"]
                              for m in MODELS)
                verdict = ("STOP_NO_ECONOMIC_INCREMENT" if not any_eco
                           else "INCONCLUSIVE_LOW_POWER")
    G = {"spec_version": "1.1.0", "conditions": res,
         "candidate": cand, "verdict": verdict}
    with open(os.path.join(RUNDIR, "GATE.json"), "w") as f:
        json.dump(G, f, indent=1)
    for m in MODELS:
        fails = [k for k, v in res[m].items()
                 if k not in ("pass_all", "folds_won_vs_market",
                              "folds_won_vs_logreg") and not v]
        print("%s pass=%s folds_vs_mkt=%d/6 vs_lr=%d/6 fail=%s" % (
            m, res[m]["pass_all"], res[m]["folds_won_vs_market"],
            res[m]["folds_won_vs_logreg"], fails))
    print("VERDICT:", verdict, "| candidate:", cand)


if __name__ == "__main__":
    main()
