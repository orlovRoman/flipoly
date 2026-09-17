"""train.py — walk-forward L0-L3 + Platt calibration (spec v1.1.0).

Folds: validate F1..F6, train on all earlier folds (SEED+F1..). Inner train
split by time: fit 70% / early-stop 10% / calibration 20%.
L3: binary objective with init_score=logit(p_market); predict raw margin,
final = sigmoid(margin + logit(p_market)) == apply_residual.
Outputs (D:\\lgbm-favorite-flip-v1\\runs\\): oof_raw.parquet, oof_cal.parquet,
models/*.txt, calibrators.json, TRAIN_MANIFEST.json.
Deterministic: num_threads=1, deterministic=True, fixed seed.
--repeat-one MODEL FOLD: retrain one cell, assert identical OOF (self-check).
"""
import argparse
import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import apply_residual, clip01, logit

OUTDIR = r"D:\lgbm-favorite-flip-v1\out"
RUNDIR = r"D:\lgbm-favorite-flip-v1\runs"
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260913
FOLD_ORDER = ["SEED", "F1", "F2", "F3", "F4", "F5", "F6"]
VAL_FOLDS = ["F1", "F2", "F3", "F4", "F5", "F6"]
MODELS = ["L0", "L1", "L2", "L3"]
CATS = ["asset", "favorite_side"]

PARAMS = {"objective": "binary", "metric": "binary_logloss",
          "learning_rate": 0.05, "num_leaves": 31,
          "min_data_in_leaf": 100, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l1": 0.0, "lambda_l2": 1.0,
          "num_threads": 1, "deterministic": True,
          "seed": SEED, "verbosity": -1}
STOP_ROUNDS = 100
MAX_ROUNDS = 2000


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prep(df, feats, cats):
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X


def train_cell(tr, va, feats, is_l3):
    cats = [c for c in CATS if c in feats]
    tr = tr.sort_values("decision_at").reset_index(drop=True)
    n = len(tr)
    i1, i2 = int(n * 0.7), int(n * 0.8)
    dfit, des, dcal = tr.iloc[:i1], tr.iloc[i1:i2], tr.iloc[i2:]
    yfit = dfit["favorite_flip"].to_numpy(float)
    yes_ = des["favorite_flip"].to_numpy(float)
    ycal = dcal["favorite_flip"].to_numpy(float)
    Xfit, Xes, Xcal = prep(dfit, feats, cats), prep(des, feats, cats), prep(dcal, feats, cats)
    Xva = prep(va, feats, cats)
    if is_l3:
        init_fit = np.array([logit(clip01(v)) for v in
                             dfit["p_market_flip"].to_numpy(float)])
        init_es = np.array([logit(clip01(v)) for v in
                            des["p_market_flip"].to_numpy(float)])
        dtr = lgb.Dataset(Xfit, label=yfit, init_score=init_fit,
                          categorical_feature=cats)
        dval = lgb.Dataset(Xes, label=yes_, init_score=init_es,
                           categorical_feature=cats, reference=dtr)
    else:
        dtr = lgb.Dataset(Xfit, label=yfit, categorical_feature=cats)
        dval = lgb.Dataset(Xes, label=yes_, categorical_feature=cats,
                           reference=dtr)
    bst = lgb.train(PARAMS, dtr, num_boost_round=MAX_ROUNDS,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(STOP_ROUNDS, verbose=False)])
    best = bst.best_iteration
    if is_l3:
        mkt_va = va["p_market_flip"].to_numpy(float)
        mkt_cal = dcal["p_market_flip"].to_numpy(float)
        raw_va = np.array([apply_residual(clip01(m), r) for m, r in zip(
            mkt_va, bst.predict(Xva, raw_score=True,
                                num_iteration=best))])
        raw_cal = np.array([apply_residual(clip01(m), r) for m, r in zip(
            mkt_cal, bst.predict(Xcal, raw_score=True,
                                 num_iteration=best))])
    else:
        raw_va = np.asarray(bst.predict(Xva, num_iteration=best),
                            dtype=float)
        raw_cal = np.asarray(bst.predict(Xcal, num_iteration=best),
                             dtype=float)
    # Platt on inner-cal slice
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(raw_cal.reshape(-1, 1), ycal)
    cal_va = lr.predict_proba(raw_va.reshape(-1, 1))[:, 1]
    return bst, best, raw_va, cal_va, (float(lr.coef_[0][0]),
                                       float(lr.intercept_[0])), len(dcal), \
        int(ycal.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat-one", nargs=2, default=None,
                    metavar=("MODEL", "FOLD"))
    a = ap.parse_args()
    os.makedirs(RUNDIR, exist_ok=True)
    os.makedirs(os.path.join(RUNDIR, "models"), exist_ok=True)
    tm = pd.read_parquet(os.path.join(OUTDIR, "TRAIN_MATRIX.parquet"))
    tm["decision_at"] = pd.to_datetime(tm["decision_at"], utc=True)
    with open(os.path.join(OUTDIR, "FEATURE_SETS.json")) as f:
        sets = json.load(f)
    y = tm["favorite_flip"].to_numpy(float)
    assert set(np.unique(y)) <= {0.0, 1.0}
    if a.repeat_one:
        models, folds = [a.repeat_one[0]], [a.repeat_one[1]]
        out_tag = "_repeatcheck"
    else:
        models, folds = MODELS, VAL_FOLDS
        out_tag = ""
    raw_rows, cal_rows, calib, blobs = [], [], {}, {}
    for model in models:
        feats = sets[model]
        is_l3 = (model == "L3")
        for fold in folds:
            tri = tm["fold"].isin(
                FOLD_ORDER[:FOLD_ORDER.index(fold)]).to_numpy()
            vai = (tm["fold"] == fold).to_numpy()
            tr, va = tm[tri].copy(), tm[vai].copy()
            assert len(tr) and len(va)
            # market wholly in one fold by construction (fold<-market_end)
            assert not set(tr["market_id"]).intersection(
                set(va["market_id"])), "market leak across folds"
            bst, best, raw_va, cal_va, (pa, pb), ncal, npos = train_cell(
                tr, va, feats, is_l3)
            key = "%s_%s" % (model, fold)
            raw_rows.append(pd.DataFrame(
                {"opportunity_id": va["opportunity_id"].values,
                 "fold": fold, "model": model, "p_raw": raw_va}))
            cal_rows.append(pd.DataFrame(
                {"opportunity_id": va["opportunity_id"].values,
                 "fold": fold, "model": model, "p_cal": cal_va}))
            calib[key] = {"method": "platt", "a": pa, "b": pb,
                          "best_iteration": int(best),
                          "n_train": int(len(tr)), "n_val": int(len(va)),
                          "n_cal": int(ncal), "n_cal_pos": int(npos)}
            blobs[key] = bst
            print("%s %s: train=%d val=%d best_it=%d flip=%.3f" % (
                model, fold, len(tr), len(va), best,
                va["favorite_flip"].mean()), flush=True)
    raw = pd.concat(raw_rows, ignore_index=True)
    cal = pd.concat(cal_rows, ignore_index=True)
    if a.repeat_one:
        prev = pd.read_parquet(os.path.join(RUNDIR, "oof_raw.parquet"))
        m = prev[(prev["model"] == models[0]) & (prev["fold"] == folds[0])]
        cur = raw
        assert len(m) == len(cur)
        same = np.allclose(m.sort_values("opportunity_id")["p_raw"].values,
                           cur.sort_values("opportunity_id")["p_raw"].values,
                           atol=0, rtol=0)
        print("REPEAT_CHECK %s_%s identical=%s" % (models[0], folds[0], same))
        assert same, "non-deterministic retrain!"
        return
    raw.to_parquet(os.path.join(RUNDIR, "oof_raw.parquet"), index=False)
    cal.to_parquet(os.path.join(RUNDIR, "oof_cal.parquet"), index=False)
    for k, b in blobs.items():
        b.save_model(os.path.join(RUNDIR, "models", "%s.txt" % k))
    isotonic_ok = all(v["n_cal"] >= 2000 and v["n_cal_pos"] >= 100
                      for v in calib.values())
    manifest = {
        "spec_version": "1.1.0", "seed": SEED, "params": PARAMS,
        "stopping_rounds": STOP_ROUNDS, "max_rounds": MAX_ROUNDS,
        "lgbm_version": lgb.__version__,
        "spec_sha256": sha_file(os.path.join(EXP, "spec.yaml")),
        "train_py_sha256": sha_file(os.path.join(EXP, "code", "train.py")),
        "feature_sets_sha256": sha_file(os.path.join(OUTDIR,
                                                     "FEATURE_SETS.json")),
        "isotonic_globally_allowed": bool(isotonic_ok),
        "calibrators": calib}
    with open(os.path.join(RUNDIR, "calibrators.json"), "w") as f:
        json.dump({k: v for k, v in calib.items()}, f, indent=1)
    with open(os.path.join(RUNDIR, "TRAIN_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    oh = sha_file(os.path.join(RUNDIR, "oof_raw.parquet"))
    print("OOF_HASH %s" % oh)
    print("isotonic_globally_allowed=%s" % isotonic_ok)


if __name__ == "__main__":
    main()
