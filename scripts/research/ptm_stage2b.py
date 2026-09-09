"""PTM stage 2 items 26-28: holdout rule set, decision log, completeness audit.

Significance: one-sided day-block recentered bootstrap (H0:E<=0) + Holm.
Review fix P1-1: NO normal approximation anywhere. Result on 11v2: ZERO
cells survive Holm at 0.05 (best adj 0.1545) -> NO significance-based
candidates. Watchlist entries below are PRE-REGISTERED for PAPER validation
on new data, explicitly NOT discoveries.
Writes holdout_rules.json, decision_log.md, completeness_audit.json.
"""
import json
import os
import sys

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)

# Load auto_findings for Holm survivors
af = json.load(open(os.path.join(BASE, "auto_findings.json"), encoding="utf-8"))

# Load stage2 for CT pairs
s2 = json.load(open(os.path.join(BASE, "ct_stage2.json"), encoding="utf-8"))

# Load paired CIs
import csv
paired = list(csv.DictReader(open(os.path.join(BASE, "ct_paired.csv"), encoding="utf-8")))

# ============================================================
# Item 26: rule set definition
# ============================================================
# Rule A: positive-CI cells with smallest Holm-adjusted perm p (watchlist).
# Nothing survives Holm at 0.05 -> status is WATCHLIST_FOR_PAPER, never
# CANDIDATE. degenerate = near-zero day variance (deep-favorite micro-edge).
rule_a = []
cands = sorted([p for p in af["positive_cells"] if p.get("holm_adj_p") is not None],
               key=lambda p: p["holm_adj_p"])[:3]
for p in cands:
    degenerate = (p.get("day_var") or 0) < 1e-3
    rule_a.append({
        "rule_id": "A_" + "_".join(str(k) for k in p["key"]),
        "rule_type": "CELL_BET",
        "asset": p["key"][2],
        "entry_rule": p["key"][1],
        "role": p["key"][3],
        "price_bin": p["key"][4],
        "n": p["n"], "pnl": p["pnl"], "ci": p["ci"],
        "p_perm": p.get("p_perm"), "holm_adj_p": p["holm_adj_p"],
        "day_var": p.get("day_var"),
        "degenerate_micro_edge": degenerate,
        "status": "WATCHLIST_FOR_PAPER",
    })

# Rule B: CT filter on T-5 (positive expectancy, CI excludes zero)
rule_b = []
for rec in paired:
    if rec["entry_rule"] == "T-5" and rec["status"] == "OK":
        ci = json.loads(rec["ct_exp_ci95"])
        if ci[0] > 0:
            rule_b.append({
                "rule_id": "B_" + rec["asset"] + "_T5_CT",
                "rule_type": "CT_FILTER",
                "asset": rec["asset"],
                "entry_rule": "T-5",
                "filter": "ct_state==REVERSION",
                "n_days": rec["days"],
                "ct_exp_ci95": ci,
                "ct_minus_c0_ci95": json.loads(rec["ct_minus_c0_ci95"]),
                "status": "CANDIDATE_FOR_HOLDOUT",
            })

# Rule C: time pairs on common-availability sample (review fix P1-4).
# All informational: no pair significant after multiplicity (best nominal
# perm_p=0.033 on T-8_vs_T-5, Holm over 3 pairs = 0.10).
rule_c = []
for p in s2["pairs"]:
    rule_c.append({
        "rule_id": "C_" + p["pair"],
        "rule_type": "CT_TIME_PAIR",
        "description": p["pair"],
        "markets": p["markets"],
        "both_ok": p.get("both_ok"),
        "mean_diff": p["mean_diff"],
        "ci95": p.get("ci95"),
        "perm_p": p.get("perm_p"),
        "status": "INFORMATIONAL",
    })

# Rule D: policy-level CT T-5 watchlist (pre-registered PAPER candidate per
# review: positive historical gross +100.10/915 with high concentration;
# robustness and post-cost profit NOT established).
t5c = s2["t5_concentration"]
rule_d = [{
    "rule_id": "D_CT_T5_ALL_ASSETS",
    "rule_type": "CT_POLICY",
    "entry_rule": "T-5",
    "filter": "ct_state==REVERSION",
    "n": t5c["n"], "gross": t5c["gross"],
    "top5": t5c["top5"], "ex_top5": t5c["ex_top5"],
    "biggest_day": t5c["biggest_day"],
    "status": "WATCHLIST_FOR_PAPER",
}]

n_surv = sum(1 for p in af["positive_cells"]
             if p.get("holm_adj_p") is not None and p["holm_adj_p"] < 0.05)
holdout = {
    "rule_set_version": "v2",
    "significance_method": af.get("method"),
    "holm_survivors_at_0_05": n_surv,
    "selection_criteria": "NO significance-based candidates (0 Holm survivors). "
                          "Watchlist = top-3 nominal cells + policy-level CT T-5, "
                          "all PRE-REGISTERED FOR PAPER validation on new data only.",
    "rules_a_cells": rule_a,
    "rules_b_ct_filter": rule_b,
    "rules_c_time_pairs": rule_c,
    "rules_d_policy": rule_d,
    "holdout_period": {
        "status": "PENDING_DATA",
        "note": "holdout period not yet defined; requires additional weeks of data or out-of-sample split",
    },
    "total_watchlist": len(rule_a) + len(rule_b) + len(rule_d),
    "total_informational": len(rule_c),
}
with open(os.path.join(BASE, "holdout_rules.json"), "w") as f:
    json.dump(holdout, f, indent=2)

# ============================================================
# Item 27: per-rule decision log
# ============================================================
decisions = []
for r in rule_a:
    decisions.append({
        "rule_id": r["rule_id"],
        "decision": "WATCHLIST_FOR_PAPER (NOT a significance-based candidate)",
        "evidence": f"perm p={r['p_perm']}, Holm adj={r['holm_adj_p']} (>=0.05: NOT significant); "
                    f"CI={r['ci']}, n={r['n']}, day_var={r['day_var']}",
        "risk": ("DEGENERATE micro-edge (day_var<1e-3): economically trivial even if real. " if r["degenerate_micro_edge"] else "")
                + "Nominal signal only; any PAPER bet requires new out-of-sample data.",
    })
for r in rule_b:
    decisions.append({
        "rule_id": r["rule_id"], "decision": "WATCHLIST_FOR_PAPER (CT expectancy CI>0 but no null test)",
        "evidence": f"CT expectancy CI={r['ct_exp_ci95']}, CI-0 diff={r['ct_minus_c0_ci95']}",
        "risk": "CI excludes zero but no multiplicity-controlled null test; CT filter depends on regime classifier.",
    })
for r in rule_c:
    decisions.append({
        "rule_id": r["rule_id"], "decision": "INFORMATIONAL_ONLY",
        "evidence": f"mean_diff={r['mean_diff']}, CI={r['ci95']}, perm_p={r['perm_p']}, markets={r['markets']} "
                    f"(both_ok={r['both_ok']})",
        "risk": "no pair significant after multiplicity (best Holm over 3 pairs = 0.10); "
                "earlier positive diffs were selection bias from conditioning on outsider status.",
    })
for r in rule_d:
    decisions.append({
        "rule_id": r["rule_id"], "decision": "WATCHLIST_FOR_PAPER (pre-registered, NOT validated)",
        "evidence": f"CT T-5 historical gross {r['gross']}/{r['n']} (+0.109/entry); top5={r['top5']}, "
                    f"ex_top5={r['ex_top5']}, biggest_day={r['biggest_day']}",
        "risk": "high concentration (92% of pnl in one day); robustness and post-cost profit NOT established; "
                "this report neither proves profitability nor gives grounds to reject.",
    })

md_lines = ["# Price-Time-Map Decision Log", "",
            f"Run: {RUN_ID}", f"Rules: {len(decisions)}", ""]
for d in decisions:
    md_lines.append(f"## {d['rule_id']}")
    md_lines.append(f"- **Decision**: {d['decision']}")
    md_lines.append(f"- **Evidence**: {d['evidence']}")
    md_lines.append(f"- **Risk**: {d['risk']}")
    md_lines.append("")

with open(os.path.join(BASE, "decision_log.md"), "w") as f:
    f.write("\n".join(md_lines))

# ============================================================
# Item 28: completeness audit
# ============================================================
audit = {
    "stage1_items_1_8": {
        "status": "DONE",
        "items": [
            "1. price_bin closed intervals: YES",
            "2. out-of-range bin: YES (plan-14-fixed+v2-out-of-range)",
            "3. SELECT ct_states with OF ... FOR UPDATE SKIP LOCKED: YES (atomically locked)",
            "4. opportunity_id key in ct_states.csv: YES",
            "5. fee separated (fee_rate column, net_pnl unknown): YES (fee_status=UNKNOWN, fee=None)",
            "6. net_pnl=None for unknown fee: YES",
            "7. timebase_mismatch=0 verified: YES",
            "8. no duplicate opportunity_ids: YES (0 dup_oid)",
        ],
    },
    "stage1_items_9_15": {
        "status": "DONE",
        "items": [
            "9. CT totals restructured to comparison_with_ct.csv (per-policy column): YES",
            "10. GRID CT n=2978 gross=-27.63 verified on 11v2 (gross, fee UNKNOWN): YES",
            "11. FUNNEL CT v2 (frozen scope): n=563 gross=-20.7471 — matches v1 exactly (same OK set, fee never applied in v1 either)",
            "12. GRID CT T-5 n=915 gross=100.10 verified on 11v2 (YES_OUTSIDER scope): YES",
            "13. T-5 top5=195 ex-top5=-94.90 verified on 11v2: YES",
            "14. positive-CI cells: v1 table=14, v2 table=15 (delta: one FUNNEL_OBSERVED cell)",
            "15. empty price_bin OK rows: v1=233 (leak), v2=0 by design (closed bins + OUT_OF_RANGE)",
        ],
    },
    "stage2_items_16_18": {
        "status": "DONE",
        "items": [
            "16. Independent universe: 14,820 markets (12,745 funnel incl 1 PENDING + 2,075 non-funnel, 5 assets)",
            "17. Bias: funnel 30 days vs non-funnel 22 days; observed_window_p50 both groups (NOT contract duration — "
            "creation time absent; 15m-membership NOT proven by this metric)",
            "18. build2 universe path fixed (was FileNotFoundError): reads _freeze/universe_<freeze>.json",
        ],
    },
    "stage2_items_19_25": {
        "status": "DONE",
        "items": [
            "19. C0/CT table per asset×entry_rule: YES (ct_time_asset.csv, 15 cells)",
            "20. Filter vs bet-reduction measured: CT-CI0 diffs in ct_paired.csv",
            "21. Paired CIs: 15 asset×rule pairs, all OK status, CT exp CI and CT-CI0 CI computed",
            "22. Time pairs REDONE on common-availability sample (no-signal -> 0, not excluded): "
            "T-12/T-8 diff=-0.0046 p=0.86; T-12/T-5 diff=+0.0106 p=0.11; T-8/T-5 diff=+0.0152 p=0.033 nominal "
            "(Holm-over-3 = 0.10, not significant). Earlier positive diffs were selection bias.",
            "23. T-5 concentration (descriptive): top5=195 (n=915), ex-top5=-94.90, biggest_day=2026-08-28 (+92.49). "
            "Described as high-concentration positive gross, NOT as proof of absence of edge.",
            "24. Winner/loser row completeness: 257+658=915 rows, all have best_ask and final_outcome: YES",
            "25. Holm over day-block recentered bootstrap null (H0:E<=0, NO normal approx): 15 positive-CI cells, "
            "0 survivors at 0.05 (best adj=0.1545). NO significance-based candidates.",
        ],
    },
    "stage2_items_26_28": {
        "status": "DONE",
        "items": [
            "26. Rule set v2: 0 significance candidates; watchlist = top-3 nominal cells + CT T-5 policy (all "
            "PRE-REGISTERED FOR PAPER on new data only); 3 time pairs informational",
            "27. Decision log: decision_log.md with corrected decisions/evidence/risks per rule",
            "28. Holdout: PENDING_DATA — no out-of-sample period defined yet",
        ],
    },
    "reviewCorrections_applied": [
        "P1-1: normal-approx Holm pseudo-p REMOVED; day-block recentered bootstrap null test + Holm",
        "P1-2(as numbered by reviewer: concentration): ex-top5/'artefact' language removed from conclusions; "
        "T-5 framed as high-concentration positive gross, robustness unestablished",
        "P1-3: universe path fixed; independent map build in progress (run ptm_20260910_indep)",
        "P1-4: pairs recomputed on availability sample with flat-for-no-signal + sign-flip perm p",
        "P2: old test_entry_selection_causality failure isolated to superseded v1 lib (decision_at unused, "
        "unchecked time_left/recorded_at consistency); v2 lib has 11 passing tests incl. boundary/late/timebase; "
        "v1 net==gross (fee never applied). verify2 locks results-preservation, not full methodology.",
        "P2: freeze chunks live on server disk (not git); meta.json with per-chunk sha256 IS committed; "
        "re-freeze from mutable DB does NOT reproduce the study — chunks must be archived.",
        "P2: execution at ask; no spread re-subtraction; fees/commissions separate unknowns.",
    ],
    "all_28_items_status": "COMPLETE_EXCEPT_HOLDOUT_AND_INDEP_MAP (see reviewCorrections_applied)",
}
with open(os.path.join(BASE, "completeness_audit.json"), "w") as f:
    json.dump(audit, f, indent=2)

print("rules_a:", len(rule_a))
print("rules_b:", len(rule_b))
print("rules_c:", len(rule_c))
print("rules_d:", len(rule_d))
print("watchlist:", holdout["total_watchlist"], "informational:", holdout["total_informational"])
print("holm survivors:", holdout["holm_survivors_at_0_05"])
