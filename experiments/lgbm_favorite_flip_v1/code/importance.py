"""importance.py — gain + OOF permutation (Holm) for L0-L3 (spec section 20).

No retrains (STOP-proportional: nothing promoted; retrain-ablation and SHAP
skipped under STOP, documented in CLOSEOUT). Writes IMPORTANCE.json.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def holm(pvals):
    """Holm step-down adjusted p-values (no scipy/statsmodels needed)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - rank))
    # enforce monotonicity along sorted order
    s = np.sort(adj[order])
    for i in range(1, m):
        s[i] = max(s[i], s[i - 1])
    out = np.empty(m)
    out[order] = s
    return out

OUTDIR = r"D:\lgbm-favorite-flip-v1\out"
RUNDIR = r"D:\lgbm-favorite-flip-v1\runs"
SEED = 20260913
REPS = 3
EPS = 1e-6
MODELS = ["L0", "L1", "L2", "L3"]
CATS = ["asset", "favorite_side"]


def clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def predict_full(bst, X, p_market=None):
    if p_market is None:
        return np.asarray(bst.predict(X), float)
    from canonical import apply_residual, clip01
    m = np.asarray([apply_residual(clip01(v), r) for v, r in zip(
        p_market, bst.predict(X, raw_score=True))])
    return m


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    tm = pd.read_parquet(os.path.join(OUTDIR, "TRAIN_MATRIX.parquet"))
    tm["decision_at"] = pd.to_datetime(tm["decision_at"], utc=True)
    sets = json.load(open(os.path.join(OUTDIR, "FEATURE_SETS.json")))
    out = {}
    for model in MODELS:
        feats = sets[model]
        is_l3 = (model == "L3")
        gains, deltas, signs = {}, {}, {}
        for fold in ["F1", "F2", "F3", "F4", "F5", "F6"]:
            va = tm[tm["fold"] == fold].copy().reset_index(drop=True)
            X = va[feats].copy()
            for c in CATS:
                if c in X.columns:
                    X[c] = X[c].astype("category")
            y = va["favorite_flip"].to_numpy(float)
            pm = (va["p_market_flip"].to_numpy(float) if is_l3 else None)
            bst = lgb.Booster(model_file=os.path.join(
                RUNDIR, "models", "%s_%s.txt" % (model, fold)))
            gi = dict(zip(bst.feature_name(),
                          bst.feature_importance(importance_type="gain")))
            for f in feats:
                gains.setdefault(f, []).append(float(gi.get(f, 0.0)))
            base = log_loss(y, clip(predict_full(bst, X, pm)))
            rng = np.random.default_rng(SEED)
            for f in feats:
                ds = []
                for r in range(REPS):
                    Xp = X.copy()
                    if f in CATS:
                        Xp[f] = pd.Categorical(
                            rng.permutation(X[f].to_numpy()),
                            categories=X[f].cat.categories)
                    else:
                        v = X[f].to_numpy(dtype=float).copy()
                        rng.shuffle(v)
                        Xp[f] = v
                    ds.append(log_loss(y, clip(predict_full(bst, Xp, pm)))
                              - base)
                ds_mean = float(np.mean(ds))
                deltas.setdefault(f, []).append(ds_mean)
            print("%s %s done" % (model, fold), flush=True)
        # Holm over features (mean delta > 0 test via sign across folds:
        # use one-sided sign test p-value per feature)
        from scipy.stats import binomtest
        feats_l = list(feats)
        pvals = []
        for f in feats_l:
            k = sum(1 for v in deltas[f] if v > 0)
            pvals.append(binomtest(k, len(deltas[f]), 0.5,
                                   alternative="greater").pvalue)
        rej = holm(pvals) < 0.05
        padj = holm(pvals)
        tab = []
        for f, pv, pa, rj in zip(feats_l, pvals, padj, rej):
            tab.append({"feature": f,
                        "mean_gain": float(np.mean(gains[f])),
                        "mean_dll": float(np.mean(deltas[f])),
                        "folds_pos": int(sum(1 for v in deltas[f] if v > 0)),
                        "p_holm": float(pa), "holm_sig": bool(rj)})
        tab.sort(key=lambda r: -r["mean_dll"])
        out[model] = tab
        sig = [t for t in tab if t["holm_sig"]]
        print("%s: %d/%d holm-sig; top: %s" % (
            model, len(sig), len(tab),
            [(t["feature"], round(t["mean_dll"], 5), t["folds_pos"])
             for t in tab[:5]]))
    with open(os.path.join(RUNDIR, "IMPORTANCE.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote IMPORTANCE.json")


if __name__ == "__main__":
    main()
