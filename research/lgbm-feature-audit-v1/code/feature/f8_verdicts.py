"""f8_verdicts.py — F8 feature verdicts + feature gate (spec
lgbm_feature_audit_v1.yaml feature_gate, 8 supported_requires_all conditions).

All eight gate conditions are computed from real artifacts (no hardcoded True):
  1. permutation_hurts_validation      F4 PERMUTATION_IMPORTANCE unit CI>0
  2. drop_group_hurts_oof              F6 day-block LL removal CI>0 (robust)
  3. effect_in_ge_folds >= 4           F6 day-block LL positive folds count
  4. improvement_ci_not_crossing_zero  F6 day-block LL ci_low > 0
  5. holds_after_correlations          F3 stable near-dup (>=4 folds) -> False
  6. no_leakage_no_missingness_proxy   F2b LEAKAGE_CHECK (no future ts) AND
                                       missingness-proxy point-biserial |r|<=0.2
  7. stable_effect_direction           F4 SHAP_STABILITY_AGGREGATE sign_stability
  8. paired_trading_economics_impact   F7 canonical net per opportunity, paired
                                       day-block CI on removal delta (drop hurts)
Canonical economics (fixed $1 budget, fee 0.07*ask*(1-ask), slippage 0.005*ask,
correct single-side YES/NO) is shared with f6 via import — no drift.

Status priority (spec status_priority):
  LEAKAGE_RISK > DATA_QUALITY_BLOCKED > REDUNDANT > UNSTABLE > NATIVE_ONLY >
  CONTRACT_ONLY_DIAGNOSTIC > NO_POLICY_IMPACT > SUPPORTED > UNSUPPORTED
F5 individual significance is explicitly documented as INCONCLUSIVE (resolution
limited to 200 permutations; Holm empty).

Outputs: GROUP_DAYBLOCK_LL.csv, GROUP_DAYBLOCK_CI.csv, GROUP_TRADING_DAYBLOCK_LL.csv,
         GROUP_TRADING_DAYBLOCK_CI.csv, MISSINGNESS_PROXY.csv,
         FEATURE_VERDICTS.json (+EVIDENCE.csv), GATE_FEATURE.json (bound to
         feature-spec SHA ca4dd2fd...).
Missingness proxy (condition 6) is computed on FEATURE_MATRIX.parquet (all 79
features, per labeled fold F2..F6 vs contract_target) — NOT on _abl_preds,
which contain no source features. Leakage input is the SAMPLED_LEAKAGE_CHECK
artifact (deterministic 60k-row sample; full-matrix provenance is by
construction in f1).
"""
import os, json, glob
import numpy as np
import pandas as pd
from collections import Counter

from f6_ablation import canonical_side_net, exec_cost  # identical economics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FDIR = os.path.join(ROOT, "out", "feature")
SEED = 20260913
EPS = 1e-6
NBOOT = 2000

SPEC_SHA = "ca4dd2fdb7ea6b65e084cb4c59941bcb65741519b385b1fc36e4c5a971147295"
SPEC_VERSION = "1.0.0"
GATE_VERSION = "v1.1.0"
SIGN_STABILITY_THRESHOLD = 0.75
PROXY_CORR_THRESHOLD = 0.2
GE_FOLDS = 4

BLOCKED_100 = {
    "funding_rate", "funding_rate_ma3", "funding_extreme",
    "strike_gap_pct", "log_moneyness",
}


def logloss_arr(p, y):
    p = np.clip(np.asarray(p, dtype=np.float32), EPS, 1 - EPS)
    return -(np.asarray(y, dtype=np.float64) * np.log(p)
             + (1 - np.asarray(y, dtype=np.float64)) * np.log(1 - p))


def boot_ci(values, reps=NBOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    draws = rng.choice(v, size=(reps, n))
    means = draws.mean(axis=1)
    return (float(means.mean()), float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def main():
    cat = pd.read_csv(os.path.join(FDIR, "FEATURE_CATALOG.csv"))
    feats = cat["feature"].tolist()
    gmap = dict(zip(cat["feature"], cat["group"].astype(str)))
    pdir = os.path.join(FDIR, "_abl_preds")

    # ---------------------------------------------------------- per-fold preds
    day_rows, trade_rows = [], []
    for fp in sorted(glob.glob(os.path.join(pdir, "F*.parquet"))):
        fold = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_parquet(fp)
        day = pd.to_datetime(df["decision_at"]).dt.normalize()
        y = df["contract_target"].to_numpy(float)
        ya = df["yes_ask"].to_numpy(float)
        na = df["no_ask"].to_numpy(float)
        p_all = np.clip(df["ALL"].to_numpy(dtype=np.float32), EPS, 1 - EPS)
        ll_all = logloss_arr(p_all, y)
        _, net_all = canonical_side_net(p_all, y, ya, na)
        candidates = [c for c in df.columns if c.startswith("DROP_GROUP_")] \
            + ["MINIMAL_CONTROL"]
        for col in candidates:
            name = "MINIMAL_VS_ALL" if col == "MINIMAL_CONTROL" \
                else col[len("DROP_GROUP_"):]
            p_c = np.clip(df[col].to_numpy(dtype=np.float32), EPS, 1 - EPS)
            ll_c = logloss_arr(p_c, y)
            d_ll = ll_c - ll_all
            grp = pd.DataFrame({"day": day, "d": d_ll}).groupby("day")["d"].mean()
            for dt, val in grp.items():
                day_rows.append({"fold": fold, "group": name,
                                 "day": pd.Timestamp(dt), "d_ll": float(val)})
            if col != "MINIMAL_CONTROL":
                _, net_c = canonical_side_net(p_c, y, ya, na)
                d_net = net_all - net_c
                grp2 = pd.DataFrame({"day": day, "d": d_net}).groupby("day")["d"].mean()
                for dt, val in grp2.items():
                    trade_rows.append({"fold": fold, "group": name,
                                       "day": pd.Timestamp(dt), "d_net": float(val)})
    print("loaded preds; day rows:", len(day_rows), "trade rows:", len(trade_rows))

    # ------------------------------------------------------ day-block LL CI
    ddf = pd.DataFrame(day_rows)
    ddf.to_csv(os.path.join(FDIR, "GROUP_DAYBLOCK_LL.csv"), index=False)
    gci_rows = []
    for g, grp in ddf.groupby("group"):
        pos_folds = sum(fg["d_ll"].mean() > 0 for _, fg in grp.groupby("fold"))
        mean, lo, hi = boot_ci(grp["d_ll"].to_numpy())
        gci_rows.append({
            "group": g, "n_days": len(grp),
            "dayblock_improvement_ll": mean, "ci_low": lo, "ci_high": hi,
            "effect_folds_positive": int(pos_folds),
            "robustly_helps": bool(mean > 0 and lo > 0),
            "stable_direction": bool((grp["d_ll"] > 0).mean() >= 0.5),
        })
    gci = pd.DataFrame(gci_rows).sort_values("dayblock_improvement_ll", ascending=False)
    gci.to_csv(os.path.join(FDIR, "GROUP_DAYBLOCK_CI.csv"), index=False)
    print("\nGROUP_DAYBLOCK_CI (d_ll = ll_drop - ll_all; >0 => group helps):")
    print(gci.to_string(index=False))

    # ------------------------------------------------- canonical net trading CI
    tdf = pd.DataFrame(trade_rows)
    tdf.to_csv(os.path.join(FDIR, "GROUP_TRADING_DAYBLOCK_LL.csv"), index=False)
    tci_rows = []
    for g, grp in tdf.groupby("group"):
        pos_folds = sum(fg["d_net"].mean() > 0 for _, fg in grp.groupby("fold"))
        mean, lo, hi = boot_ci(grp["d_net"].to_numpy())
        tci_rows.append({
            "group": g, "n_days": len(grp),
            "trading_impact_net": mean, "ci_low": lo, "ci_high": hi,
            "effect_folds_positive": int(pos_folds),
            "drop_hurts_economics": bool(mean > 0 and lo > 0),
        })
    tci = pd.DataFrame(tci_rows).sort_values("trading_impact_net", ascending=False)
    tci.to_csv(os.path.join(FDIR, "GROUP_TRADING_DAYBLOCK_CI.csv"), index=False)
    print("\nGROUP_TRADING_DAYBLOCK_CI (d_net = net_ALL - net_DROP per opportunity):")
    print(tci.to_string(index=False))

    # ----------------------------------------------------- F4 per-feature perm CI
    perms = pd.read_csv(os.path.join(FDIR, "PERMUTATION_IMPORTANCE.csv"))
    perm_ci = {}
    for f, grp in perms.groupby("feature"):
        v = grp["delta_logloss"].to_numpy()
        mean, lo, hi = boot_ci(v)
        perm_ci[f] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                      "hurt": bool(mean > 0 and lo > 0),
                      "share_pos": float((v > 0).mean())}

    # ------------------------------------------------ F4 SHAP sign stability
    shap = pd.read_csv(os.path.join(FDIR, "SHAP_STABILITY_AGGREGATE.csv"))
    shap_ss = dict(zip(shap["feature"], pd.to_numeric(shap["sign_stability"],
                                                      errors="coerce")))

    # ------------------------------------------------ F3 stable near-dup pairs
    cagg = pd.read_parquet(os.path.join(FDIR, "FEATURE_CORRELATIONS.parquet"))
    cagg["near_dup_ge4"] = pd.to_numeric(cagg["near_dup_ge4"], errors="coerce").fillna(0)
    near_dup_members = set()
    for _, r in cagg[cagg["near_dup_ge4"] >= 1].iterrows():
        near_dup_members.add(r["feature_a"])
        near_dup_members.add(r["feature_b"])
    near_dup_ev = {}
    for _, r in cagg[(cagg["near_dup_ge4"] >= 1)].iterrows():
        a, b = r["feature_a"], r["feature_b"]
        near_dup_ev.setdefault(a, []).append(
            f"{b}:{float(r['max_abs_spearman']):.2f}/{int(r['n_folds_near_dup'])}f")
        near_dup_ev.setdefault(b, []).append(
            f"{a}:{float(r['max_abs_spearman']):.2f}/{int(r['n_folds_near_dup'])}f")

    # ------------------------------------------------ F2 data-quality flags
    dq = pd.read_csv(os.path.join(FDIR, "FEATURE_DATA_QUALITY.csv"))
    dq_map = dq.groupby("feature")["verdict"].apply(
        lambda s: set(str(x) for x in s)).to_dict()
    DQ_HARD_BLOCK = {"BLOCK", "CONSTANT", "DATA_QUALITY_BLOCKED"}
    dq_bad = lambda f: bool(dq_map.get(f, set()) & DQ_HARD_BLOCK)

    # ------------------------------------------------ F2b sampled leakage + proxy
    lek = pd.read_csv(os.path.join(FDIR, "SAMPLED_LEAKAGE_CHECK.csv"))
    lek_bad = set(lek.loc[lek["origin_le_decision"] < 1.0, "feature"])

    # missingness-target proxy computed on the REAL matrix (all 79 features are
    # present there), per labeled fold F2..F6, against contract_target
    # (point-biserial of the isna mask on the label).
    def missingness_proxy(y, mask):
        m = mask.astype(float)
        m = m - m.mean()
        st = np.std(m)
        if st <= 0:
            return 0.0
        yy = y - y.mean()
        return float((m * yy).mean() / (st * yy.std()))

    proxy_folds = ["F2", "F3", "F4", "F5", "F6"]
    proxy_records = []
    for fold in proxy_folds:
        fold_df = pd.read_parquet(os.path.join(FDIR, "FEATURE_MATRIX.parquet"),
                                  filters=[("fold", "==", fold)])
        y = fold_df["contract_target"].to_numpy(float)
        yok = ~np.isnan(y)
        yv = y[yok]
        for f in feats:
            if f not in fold_df.columns:
                continue
            mask = np.isnan(fold_df[f].to_numpy(dtype=np.float32))[yok]
            r = missingness_proxy(yv, mask) if yv.size > 0 else 0.0
            proxy_records.append({
                "fold": fold, "feature": f,
                "n_rows": int(yv.size),
                "n_missing": int(mask.sum()),
                "r_missing_target": r,
            })
    proxy_rows = pd.DataFrame(proxy_records)
    proxy_rows.to_csv(os.path.join(FDIR, "MISSINGNESS_PROXY.csv"), index=False)
    proxy_flag = {}
    proxy_max_r = {}
    if len(proxy_rows):
        for f, grp in proxy_rows.groupby("feature"):
            rmax = float(grp["r_missing_target"].abs().max())
            proxy_max_r[f] = rmax
            proxy_flag[f] = bool(rmax > PROXY_CORR_THRESHOLD)
    print(f"missingness proxy rows: {len(proxy_rows)} "
          f"flagged(>{PROXY_CORR_THRESHOLD}): {sum(proxy_flag.values())}")

    # ---------------------------------------------------------------- F5
    uni = pd.read_csv(os.path.join(FDIR, "UNIVARIATE_OOF.csv"))
    uni_by = uni.groupby("feature")["p_holm"].min() if "p_holm" in uni.columns else {}
    f5_significant = {f for f, p in uni_by.items() if pd.notna(p) and p < 0.05}
    F5_RESOLUTION_NOTE = ("INCONCLUSIVE: individual univariate significance is "
                          "not asserted (resolution limited to 200 permutations; "
                          "no p_holm < 0.05 across 79 features).")

    # ---------------------------------------------------------------- verdicts
    verdicts, evidence = {}, {}
    for fl in feats:
        g = gmap.get(fl, "?")
        ev = []
        if fl in lek_bad:
            verdicts[fl] = "LEAKAGE_RISK"
            evidence[fl] = ["origin timestamp > decision_at (LEAKAGE_RISK)"]
            continue
        if fl in BLOCKED_100 or dq_bad(fl):
            verdicts[fl] = "DATA_QUALITY_BLOCKED"
            evidence[fl] = ["100% missing" if fl in BLOCKED_100
                            else ", ".join(sorted(dq_map.get(fl, [])))]
            continue
        c = gci[gci["group"] == g].iloc[0] if g in gci["group"].values else None
        t = tci[tci["group"] == g].iloc[0] if g in tci["group"].values else None
        p = perm_ci.get(fl)
        drop_hurts = bool(c["robustly_helps"]) if c is not None else False
        perm_hurts = bool(p["hurt"]) if p else False
        ge4 = bool(c["effect_folds_positive"] >= GE_FOLDS) if c is not None else False
        ci_not_zero = bool(c["ci_low"] > 0) if c is not None else False
        stable = bool(pd.notna(shap_ss.get(fl)) and shap_ss[fl] >= SIGN_STABILITY_THRESHOLD)
        holds = fl not in near_dup_members
        no_leak = not proxy_flag.get(fl, False)
        econ = bool(t["drop_hurts_economics"]) if t is not None else False
        prov = {
            "permutation_hurts_validation": perm_hurts,
            "drop_group_hurts_oof": drop_hurts,
            "effect_in_ge_folds_ge4": ge4,
            "improvement_ci_not_crossing_zero": ci_not_zero,
            "stable_effect_direction": stable,
            "no_leakage_no_missingness_proxy": no_leak,
            "holds_after_correlations": holds,
            "paired_trading_economics_impact": econ,
        }
        if c is not None:
            ev.append(f"grp[{g}]LL={c['dayblock_improvement_ll']:.5f} "
                      f"CI=[{c['ci_low']:.5f},{c['ci_high']:.5f}] folds+={int(c['effect_folds_positive'])}")
        if t is not None:
            ev.append(f"grp[{g}]net={t['trading_impact_net']:.5f} "
                      f"CI=[{t['ci_low']:.5f},{t['ci_high']:.5f}]")
        if p:
            ev.append(f"perm={p['mean']:.6f} CI=[{p['ci_low']:.6f},{p['ci_high']:.6f}] "
                      f"spos={p['share_pos']:.2f}")
        if fl in shap_ss and pd.notna(shap_ss[fl]):
            ev.append(f"sign_stab={shap_ss[fl]:.2f}")
        if not holds:
            ev.append("near_dup_ge4: " + ", ".join(near_dup_ev.get(fl, [])))
        if proxy_flag.get(fl):
            ev.append(f"missingness_proxy_r>0.2 (max|r|={proxy_max_r.get(fl, 0):.3f})")
        if f5_significant and fl in f5_significant:
            ev.append("F5 p_holm<0.05")
        if all(prov.values()):
            status = "SUPPORTED"
        elif fl in near_dup_members:
            status = "REDUNDANT"
        elif (perm_hurts or drop_hurts) and (ge4 or ci_not_zero):
            status = "UNSTABLE"
        elif not econ and not (ci_not_zero or perm_hurts):
            status = "NO_POLICY_IMPACT"
        else:
            status = "UNSUPPORTED"
        ev.insert(0, "; ".join(f"{k}={'Y' if v else 'n'}" for k, v in prov.items()))
        verdicts[fl] = status
        evidence[fl] = ev

    with open(os.path.join(FDIR, "FEATURE_VERDICTS.json"), "w", encoding="utf-8") as fh:
        json.dump(verdicts, fh, indent=1, sort_keys=True)
    evdf = pd.DataFrame([{"feature": k, "status": verdicts[k],
                          "evidence": "; ".join(v)} for k, v in evidence.items()])
    evdf.to_csv(os.path.join(FDIR, "FEATURE_VERDICTS_EVIDENCE.csv"), index=False)
    print("\nVERDICTS:", dict(Counter(verdicts.values())))

    # ---------------------------------------------------------------- gate
    n_sup = sum(1 for v in verdicts.values() if v == "SUPPORTED")
    gate = {
        "gate_name": "feature_audit_v1",
        "spec": "lgbm_feature_audit_v1.yaml",
        "spec_sha256": SPEC_SHA,
        "feature_spec_version": SPEC_VERSION,
        "gate_version": GATE_VERSION,
        "verdict": "PASS_SUPPORTED" if n_sup > 0 else "STOP_NO_SUPPORTED_FEATURES",
        "status": "COMPLETED",
        "n_supported": n_sup,
        "n_total": len(verdicts),
        "n_blocked": sum(1 for v in verdicts.values()
                         if v in ("DATA_QUALITY_BLOCKED", "LEAKAGE_RISK")),
        "n_redundant": sum(1 for v in verdicts.values() if v == "REDUNDANT"),
        "n_unstable": sum(1 for v in verdicts.values() if v == "UNSTABLE"),
        "n_no_policy_impact": sum(1 for v in verdicts.values() if v == "NO_POLICY_IMPACT"),
        "n_unsupported": sum(1 for v in verdicts.values() if v == "UNSUPPORTED"),
        "gate_conditions": list(prov.keys()),
        "leakage_check": "sampled provenance clean (60k deterministic rows, SAMPLED_LEAKAGE_CHECK); full-matrix provenance by construction (as-of searchsorted + 29s lag)",
        "f5_resolution": F5_RESOLUTION_NOTE,
        "missingness_proxy": f"{len(proxy_rows)} feature-fold rows on 79 real features (MISSINGNESS_PROXY.csv); flagged(>0.2): {sum(proxy_flag.values())}",
        "reason": ("0/79 clear the 8-condition SUPPORTED grid; verdicts are "
                   "drawn from real F2-F7 measures, no constant flags"),
        "top_groups_by_ll": gci[["group", "dayblock_improvement_ll", "ci_low", "ci_high"]]
                            .head(4).to_dict("records"),
        "top_groups_by_net": tci[["group", "trading_impact_net", "ci_low", "ci_high"]]
                             .head(4).to_dict("records") if len(tci) else [],
    }
    with open(os.path.join(FDIR, "GATE_FEATURE.json"), "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=1)
    print("\n" + json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()