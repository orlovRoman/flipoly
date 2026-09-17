"""evaluate.py — forecast metrics, paired comparisons, slices (spec v1.1.0).

Reads TRAIN_MATRIX.parquet + oof_raw/oof_cal. Writes METRICS.json and
OOF_FULL.parquet (joined frame for economics/gate) to runs/.
Bootstrap: by day and by market (market kept whole with all windows).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

OUTDIR = r"D:\lgbm-favorite-flip-v1\out"
RUNDIR = r"D:\lgbm-favorite-flip-v1\runs"
SEED = 20260913
REPS = 2000
EPS = 1e-6
MODELS = ["L0", "L1", "L2", "L3"]
PAIRS = [("L0", "market"), ("L1", "market"), ("L2", "market"),
         ("L3", "market"), ("L0", "logreg"), ("L1", "logreg"),
         ("L2", "logreg"), ("L3", "logreg"), ("L3", "L1"), ("L2", "L1")]


def clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def ece_score(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1] if i < bins - 1
                              else p <= edges[i + 1] + 1e-12)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def slope_intercept(y, p):
    x = np.log(clip(p) / (1 - clip(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def metrics_block(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = {"n": int(len(y)), "flip_rate": float(y.mean()),
           "logloss": float(log_loss(y, clip(p))),
           "brier": float(brier_score_loss(y, p)),
           "ece": ece_score(y, p)}
    s, b = slope_intercept(y, p)
    out["slope"] = s
    out["intercept"] = b
    if len(np.unique(y)) == 2 and len(y) > 10:
        out["auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["auc"] = None
        out["pr_auc"] = None
    return out


def boot_delta(df, col_a, col_b, unit, reps=REPS, seed=SEED):
    """Mean(a-b) with percentile CI resampling whole units."""
    rng = np.random.default_rng(seed)
    units = df[unit].to_numpy()
    a = df[col_a].to_numpy(float)
    b = df[col_b].to_numpy(float)
    d = a - b
    uk = np.unique(units)
    idx = {u: np.nonzero(units == u)[0] for u in uk}
    uk = np.asarray(uk)
    est = float(d.mean())
    boots = np.empty(reps)
    for r in range(reps):
        pick = uk[rng.integers(0, len(uk), len(uk))]
        sel = np.concatenate([idx[u] for u in pick])
        boots[r] = d[sel].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi)}


def ll_col(y, p):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    tm = pd.read_parquet(os.path.join(OUTDIR, "TRAIN_MATRIX.parquet"))
    raw = pd.read_parquet(os.path.join(RUNDIR, "oof_raw.parquet"))
    cal = pd.read_parquet(os.path.join(RUNDIR, "oof_cal.parquet"))
    wide_raw = raw.pivot(index="opportunity_id", columns="model",
                         values="p_raw")
    wide_cal = cal.pivot(index="opportunity_id", columns="model",
                         values="p_cal")
    df = tm.merge(wide_raw.add_prefix("raw_"), left_on="opportunity_id",
                  right_index=True, how="inner")
    df = df.merge(wide_cal.add_prefix("cal_"), left_on="opportunity_id",
                  right_index=True, how="inner")
    assert (df["fold"] != "SEED").all()
    df["day"] = pd.to_datetime(df["decision_at"], utc=True).dt.strftime(
        "%Y-%m-%d")
    df["ll_market"] = ll_col(df["favorite_flip"].values,
                             df["p_market_flip"].values)
    lg = df["logreg_available"].to_numpy(bool)
    df["ll_logreg"] = np.nan
    df.loc[lg, "ll_logreg"] = ll_col(df.loc[lg, "favorite_flip"].values,
                                     df.loc[lg, "p_logreg_flip"].values)
    for m in MODELS:
        df["ll_raw_" + m] = ll_col(df["favorite_flip"].values,
                                   df["raw_" + m].values)
        df["ll_cal_" + m] = ll_col(df["favorite_flip"].values,
                                   df["cal_" + m].values)
    df.to_parquet(os.path.join(RUNDIR, "OOF_FULL.parquet"), index=False)

    M = {"controls": {}, "models_raw": {}, "models_cal": {},
         "paired": {}, "slices": {}, "per_fold": {}}
    M["controls"]["market"] = metrics_block(df["favorite_flip"],
                                            df["p_market_flip"])
    M["controls"]["logreg"] = metrics_block(df.loc[lg, "favorite_flip"],
                                            df.loc[lg, "p_logreg_flip"])
    M["controls"]["logreg_coverage"] = {
        "frac": float(lg.mean()), "n": int(lg.sum()),
        "markets": int(df.loc[lg, "market_id"].nunique())}
    for m in MODELS:
        M["models_raw"][m] = metrics_block(df["favorite_flip"],
                                           df["raw_" + m])
        M["models_cal"][m] = metrics_block(df["favorite_flip"],
                                           df["cal_" + m])
        pf = {}
        for f, g in df.groupby("fold"):
            pf[f] = {"logloss": metrics_block(g["favorite_flip"],
                                              g["cal_" + m])["logloss"],
                     "n": int(len(g))}
        M["per_fold"][m] = pf
    # per-fold controls
    pf_mkt, pf_lr = {}, {}
    for f, g in df.groupby("fold"):
        pf_mkt[f] = metrics_block(g["favorite_flip"],
                                  g["p_market_flip"])["logloss"]
        gg = g[g["logreg_available"]]
        pf_lr[f] = (metrics_block(gg["favorite_flip"],
                                  gg["p_logreg_flip"])["logloss"]
                    if len(gg) else None)
    M["per_fold"]["market"] = pf_mkt
    M["per_fold"]["logreg"] = pf_lr

    ctrl_col = {"market": "ll_market", "logreg": "ll_logreg",
                "L1": "ll_cal_L1"}
    for (m, c) in PAIRS:
        sub = df
        if c == "logreg":
            sub = df[df["logreg_available"]]
        a, b = "ll_cal_" + m, ctrl_col[c]
        s = sub[["day", "market_id", a, b]].dropna()
        M["paired"]["%s_vs_%s" % (m, c)] = {
            "n": int(len(s)),
            "markets": int(s["market_id"].nunique()),
            "flips": int(sub["favorite_flip"].sum()),
            "by_day": boot_delta(s, b, a, "day"),
            "by_market": boot_delta(s, b, a, "market_id")}
    # slices on calibrated
    for col, name in (("decision_window", "window"),
                      ("favorite_side", "side"), ("asset", "asset"),
                      ("fold", "fold")):
        sl = {}
        for k, g in df.groupby(col):
            d = {"n": int(len(g)),
                 "flip_rate": float(g["favorite_flip"].mean())}
            for m in MODELS + ["market"]:
                p = g["cal_" + m] if m in MODELS else g["p_market_flip"]
                d[m] = {"logloss": float(log_loss(g["favorite_flip"],
                                                  clip(p))),
                        "brier": float(brier_score_loss(g["favorite_flip"],
                                                        p))}
            gg = g[g["logreg_available"]]
            d["logreg"] = {"logloss": float(log_loss(
                gg["favorite_flip"], clip(gg["p_logreg_flip"]))),
                "n": int(len(gg))} if len(gg) else None
            sl[str(k)] = d
        M["slices"][name] = sl
    # price bands (in-band vs out-of-band)
    df["band"] = pd.cut(df["outsider_mid"],
                        bins=[0.05, 0.10, 0.20, 0.30, 0.50],
                        right=False, include_lowest=True).astype(str)
    bl = {}
    for k, g in df.groupby("band"):
        bl[str(k)] = {"n": int(len(g)),
                      "flip_rate": float(g["favorite_flip"].mean()),
                      "mkt_ll": float(log_loss(g["favorite_flip"],
                                               clip(g["p_market_flip"]))),
                      "L3_ll": float(log_loss(g["favorite_flip"],
                                              clip(g["cal_L3"])))}
    bo = df[df["out_of_band"]]
    bl["OUT_OF_BAND"] = {"n": int(len(bo)),
                         "flip_rate": float(bo["favorite_flip"].mean())
                         if len(bo) else None}
    M["slices"]["band"] = bl
    with open(os.path.join(RUNDIR, "METRICS.json"), "w") as f:
        json.dump(M, f, indent=1, default=str)
    print("=== pooled calibrated logloss ===")
    for m in MODELS:
        print("%s %.4f (brier %.4f auc %s)" % (
            m, M["models_cal"][m]["logloss"], M["models_cal"][m]["brier"],
            M["models_cal"][m]["auc"]))
    print("market %.4f | logreg %.4f (cov %.3f)" % (
        M["controls"]["market"]["logloss"],
        M["controls"]["logreg"]["logloss"],
        M["controls"]["logreg_coverage"]["frac"]))
    print("=== paired LL improvement (est [lo,hi] by_market) ===")
    for k, v in M["paired"].items():
        print("%s n=%d: %.4f [%.4f, %.4f]" % (
            k, v["n"], v["by_market"]["est"], v["by_market"]["lo"],
            v["by_market"]["hi"]))


if __name__ == "__main__":
    main()
