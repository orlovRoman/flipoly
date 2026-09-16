"""f4_importance.py — F4 frozen-blob importance audit (spec lgbm_feature_audit_v1.yaml).

Frozen blobs only, NO refit. For each ACTIVE frozen model, evaluate on its own
deployment rows (shared_ds model_slot + slot_version) across validation folds.
  - lgbm: split & gain importances from the frozen Booster (+ SHAP TreeExplainer
    sign stability + in-stratum permutation importance).
  - logreg: absolute |coef| mapped onto catalog feature names (supplementary).
Permutation shuffles within strata [day, asset, regime] and only within one fold
(never shuffling across periods). Metric = logloss and brier vs contract_target.
Aggregation weights each (model, fold) unit by deployment row volume (n).

Outputs: out/feature/MODEL_IMPORTANCE.csv, SHAP_STABILITY.csv,
         PERMUTATION_IMPORTANCE.csv
"""
import os
import hashlib
import base64
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATRIX = os.path.join(ROOT, "out", "feature", "FEATURE_MATRIX.parquet")
SHARED = os.path.join(ROOT, "out", "shared_ds", "shared_dataset.parquet")
INV = os.path.join(ROOT, "out", "MODEL_INVENTORY.csv")
BLOBS = os.path.join(ROOT, "data", "blobs.csv")
OUT_DIR = os.path.join(ROOT, "out", "feature")

EVAL_FOLDS = ["F2", "F3", "F4", "F5", "F6"]
K_SAMPLE = 2000
B_PERM = 3
SEED = 20260916


def rng_for(model_id, fold):
    h = hashlib.md5(f"{model_id}|{fold}|{SEED}".encode()).hexdigest()
    return np.random.default_rng(int(h[:12], 16))


def proba_of(mdl, X):
    if isinstance(mdl, lgb.Booster):
        return np.clip(mdl.predict(X), 1e-6, 1 - 1e-6)
    if hasattr(mdl, "predict_proba"):
        p = mdl.predict_proba(X)
        return np.clip(p[:, 1] if p.ndim > 1 else p, 1e-6, 1 - 1e-6)
    return np.clip(mdl.predict(X), 1e-6, 1 - 1e-6)


def base_booster(mdl):
    if isinstance(mdl, lgb.Booster):
        return mdl
    if hasattr(mdl, "raw_model"):
        raw = mdl.raw_model
        if hasattr(raw, "booster_"):
            return raw.booster_
    if hasattr(mdl, "estimator") and isinstance(mdl.estimator, lgb.Booster):
        return mdl.estimator
    if hasattr(mdl, "booster_"):
        return mdl.booster_
    if hasattr(mdl, "estimators_") and hasattr(mdl.estimators_, "predict"):
        return None
    if hasattr(mdl, "feature_names_in_"):
        return None
    return mdl


def logloss_brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    br = float(((p - y) ** 2).mean())
    return ll, br


def load_blobs():
    import csv
    csv.field_size_limit(2_000_000_000)
    out = {}
    with open(BLOBS, "r", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            payload = "".join(row[1].split())
            out[int(row[0])] = pickle.loads(base64.b64decode(payload))
    return out


def main():
    inv = pd.read_csv(INV, dtype=str, keep_default_na=False)
    act = inv[inv["is_active"] == "True"].copy()
    blobs = load_blobs()

    mat = pd.read_parquet(MATRIX)
    sd = pd.read_parquet(SHARED, columns=["decision_event_id", "model_slot", "slot_version"])
    sd["slot_version"] = sd["slot_version"].astype(str).str.replace(r"\.0$", "", regex=True)
    m = mat.merge(sd, on="decision_event_id", how="left")

    catalog_cols = list(mat.columns[8:])  # after 8 target cols
    rows_model = []     # MODEL_IMPORTANCE.csv
    rows_shap = []      # SHAP_STABILITY.csv
    rows_perm = []      # PERMUTATION_IMPORTANCE.csv

    for _, r in act.iterrows():
        mid = int(r["id"])
        slot = str(r["asset"])
        ver = str(r["version"])
        mtype = r["model_type"]
        feats = [f for f in str(r["features"]).split(",") if f]

        rows = m[(m["model_slot"] == slot) & (m["slot_version"] == ver) &
                 m["fold"].isin(EVAL_FOLDS)]
        if len(rows) == 0:
            continue
        present = [f for f in feats if f in m.columns]
        used = [f for f in feats if f in present]

        if mtype == "lgbm":
            mdl = blobs[mid]
            booster = base_booster(mdl)
            if booster is None:
                continue
            split_imp = dict(zip(booster.feature_name(), booster.feature_importance("split")))
            gain_imp = dict(zip(booster.feature_name(), booster.feature_importance("gain")))
            explainer = shap.TreeExplainer(booster)
            per_fold = {}
            for fold in EVAL_FOLDS:
                fr = rows[rows["fold"] == fold]
                if len(fr) == 0:
                    continue
                y = fr["contract_target"].to_numpy(dtype=float)
                X = fr[used].to_numpy(dtype=np.float32)
                keep = np.isfinite(y)
                if keep.sum() == 0:
                    continue
                y, X = y[keep], X[keep]
                if len(X) > K_SAMPLE:
                    rng = rng_for(mid, fold)
                    ix = rng.choice(len(X), K_SAMPLE, replace=False)
                    y, X = y[ix], X[ix]
                n = len(X)

                # SHAP sign per feature within this fold
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                sv = np.asarray(sv)
                shap_sign = {}
                for j, f in enumerate(used):
                    if np.isfinite(sv[:, j]).any():
                        mval = float(np.nanmean(sv[:, j]))
                    else:
                        mval = np.nan
                    shap_sign[f] = mval

                # base metric
                p = proba_of(mdl, X)
                base_ll, base_br = logloss_brier(y, p)

                # permutation within strata (day x asset x regime), same fold
                d = fr["decision_at"].dt.date.astype(str).astype(object).to_numpy()[keep]
                if len(d) > len(X):  # subsample stratum labels in lockstep
                    d = d[ix]
                strata = pd.factorize(np.array([f"{a}|{b}|{c}" for a, b, c in
                                       zip(d, [slot] * len(d), [r["model_regime"]] * len(d))]))[0]
                perm = {}
                for j, f in enumerate(used):
                    if not np.isfinite(X[:, j]).any():
                        perm[f] = (np.nan, np.nan)
                        continue
                    dlls, dbrs = [], []
                    for _ in range(B_PERM):
                        rng = rng_for(f"{mid}|{fold}|{f}", _)
                        Xp = X.copy()
                        for s in np.unique(strata):
                            sel = np.where(strata == s)[0]
                            if len(sel) < 2:
                                continue
                            Xp[sel, j] = rng.permutation(X[sel, j])
                        pp = proba_of(mdl, Xp)
                        lla, bra = logloss_brier(y, pp)
                        dlls.append(lla - base_ll)
                        dbrs.append(bra - base_br)
                    perm[f] = (float(np.mean(dlls)), float(np.mean(dbrs)))

                per_fold[fold] = {"n": n,
                                  "shap_sign": shap_sign,
                                  "perm": perm}
                for f in used:
                    rows_shap.append({"feature": f, "fold": fold, "model_id": mid,
                                      "asset": slot, "version": ver,
                                      "n": n, "mean_shap": shap_sign.get(f, np.nan),
                                      "abs_shap": abs(shap_sign.get(f, np.nan))})
                    dll, dbr = perm[f]
                    rows_perm.append({"feature": f, "fold": fold, "model_id": mid,
                                      "asset": slot, "version": ver, "n": n,
                                      "delta_logloss": dll, "delta_brier": dbr})

            for f in used:
                s = split_imp.get(f, 0.0)
                g = gain_imp.get(f, 0.0)
                n_units = len(per_fold)
                rows_model.append({"model_id": mid, "asset": slot, "version": ver,
                                   "model_type": "lgbm", "feature": f,
                                   "split_importance": float(s), "gain_importance": float(g),
                                   "n_fold_units": n_units})
        else:  # logreg -> |coef| supplementary
            mdl = blobs[mid]
            c = getattr(mdl, "coef_", None)
            if c is None or not used:
                continue
            c = np.abs(np.asarray(c)).ravel() if c.ndim > 1 else np.abs(np.asarray(c))
            n_fold = int(rows["fold"].nunique())
            for j, f in enumerate(used):
                if j < len(c):
                    rows_model.append({"model_id": mid, "asset": slot, "version": ver,
                                       "model_type": "logreg", "feature": f,
                                       "split_importance": np.nan, "gain_importance": np.nan,
                                       "n_fold_units": n_fold})

    # ---------------- write outputs ----------------
    dfm = pd.DataFrame(rows_model, columns=["model_id", "asset", "version", "model_type",
                                            "feature", "split_importance", "gain_importance",
                                            "n_fold_units"])
    dfs = pd.DataFrame(rows_shap, columns=["feature", "fold", "model_id", "asset", "version",
                                           "n", "mean_shap", "abs_shap"])
    dfp = pd.DataFrame(rows_perm, columns=["feature", "fold", "model_id", "asset", "version",
                                           "n", "delta_logloss", "delta_brier"])
    if len(dfm):
        dfm.to_csv(os.path.join(OUT_DIR, "MODEL_IMPORTANCE_DETAIL.csv"), index=False)
    if len(dfs):
        dfs.to_csv(os.path.join(OUT_DIR, "SHAP_STABILITY.csv"), index=False)
    if len(dfp):
        dfp.to_csv(os.path.join(OUT_DIR, "PERMUTATION_IMPORTANCE.csv"), index=False)
    print("MODEL_IMPORTANCE_DETAIL rows:", len(dfm))
    print("SHAP_STABILITY rows:", len(dfs))
    print("PERMUTATION_IMPORTANCE rows:", len(dfp))

    # ---------------- feature-level aggregates ----------------
    feat_all = list(catalog_cols)
    mu = act.copy()
    mu["uses"] = mu["features"].map(lambda s: set(s.split(",")) if s else set())
    model_uses = {f: int(sum(f in x for x in mu["uses"])) for f in feat_all}

    mi = pd.DataFrame({"feature": feat_all})
    if len(dfm):
        wz = dfm[dfm["model_type"] == "lgbm"].groupby("feature").agg(
            n_models=("model_id", "nunique"),
            n_fold_units=("n_fold_units", "sum"),
            split_sum=("split_importance", "sum"),
            gain_sum=("gain_importance", "sum"),
        ).reset_index()
        mi = mi.merge(wz, on="feature", how="left")
    else:
        mi["n_models"] = 0
        mi["n_fold_units"] = 0
        mi["split_sum"] = 0.0
        mi["gain_sum"] = 0.0
    mi["model_uses"] = mi["feature"].map(model_uses).fillna(0).astype(int)
    mi["used_in_eval_models"] = (mi["n_models"] > 0).astype(int)
    mi = mi.sort_values(["model_uses", "gain_sum"], ascending=[False, False])
    mi.to_csv(os.path.join(OUT_DIR, "MODEL_IMPORTANCE.csv"), index=False)
    print("\nMODEL_IMPORTANCE.csv (top-15):")
    print(mi.head(15).fillna("").to_string(index=False))

    if len(dfs):
        dfs_ = dfs.dropna(subset=["mean_shap"]).copy()
        sgn = np.sign(dfs_["mean_shap"])
        sign_stat = dfs_.groupby("feature")["mean_shap"].agg(
            n_neg=lambda s: int((np.sign(s) < 0).sum()),
            n_pos=lambda s: int((np.sign(s) > 0).sum()),
        )
        wabs = dfs_["abs_shap"] * dfs_["n"]
        wsum = dfs_.groupby("feature")["n"].sum()
        was = wabs.groupby(dfs_["feature"]).sum() / wsum
        agg = pd.DataFrame({
            "n_units": dfs_.groupby("feature")["mean_shap"].count(),
            "n_rows": wsum,
            "weighted_abs_shap": was,
            "mean_shap": dfs_.groupby("feature")["mean_shap"].mean(),
            "n_neg_units": sign_stat["n_neg"],
            "n_pos_units": sign_stat["n_pos"],
        }).reset_index()
        agg["sign_stability"] = agg.apply(
            lambda r: max(r["n_pos_units"], r["n_neg_units"]) / (r["n_pos_units"] + r["n_neg_units"])
            if (r["n_pos_units"] + r["n_neg_units"]) > 0 else np.nan, axis=1)
        agg["sign"] = np.where(agg["mean_shap"] > 0, "+", np.where(agg["mean_shap"] < 0, "-", ""))
        agg = agg.sort_values("weighted_abs_shap", ascending=False)
        agg.to_csv(os.path.join(OUT_DIR, "SHAP_STABILITY_AGGREGATE.csv"), index=False)
        print("\nSHAP top-12 (weighted |shap|, sign_stability):")
        print(agg.head(12)[["feature", "weighted_abs_shap", "sign", "sign_stability"]].to_string(index=False))
    if len(dfp):
        per = dfp.dropna(subset=["delta_logloss"]).copy()
        wg = per.groupby("feature")["n"].transform("sum")
        agg_p = per.groupby("feature").apply(
            lambda g: pd.Series({
                "n_units": len(g),
                "n_rows": int(g["n"].sum()),
                "delta_logloss": float(np.average(g["delta_logloss"], weights=g["n"])),
                "delta_brier": float(np.average(g["delta_brier"], weights=g["n"])),
            }), include_groups=False).reset_index()
        dpos = per.groupby("feature").apply(
            lambda g: float((g["delta_logloss"] > 0).mean()), include_groups=False).to_dict()
        agg_p["share_positive"] = agg_p["feature"].map(dpos)
        agg_p = agg_p.sort_values("delta_logloss", ascending=False)
        print("\nPERMUTATION top-12 (delta_logloss):")
        print(agg_p.head(12).to_string(index=False))
        agg_p.to_csv(os.path.join(OUT_DIR, "PERMUTATION_AGGREGATE.csv"), index=False)


if __name__ == "__main__":
    main()