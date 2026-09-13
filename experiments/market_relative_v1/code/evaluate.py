"""evaluate.py v1.0.0 — purged walk-forward (spec v1.0.4, T16-T17).

Folds (expanding, 1d embargo, 7d validation, assignment by canonical end):
  F1 train [07-06,07-26) val [07-27,08-03) ... (see FOLDS)
Models: baseline(market-only) + linear + lgbm A/B/C (4 configs, frozen).
Train rows time-ordered; ES split inside train (train.py).
OOF pooled per model -> Wilson bins (frozen for that model's signals).
Economics: historical top-of-book + 0.5% adverse (ev.realized_hist).
Outputs: oof.csv, bins.json, metrics.json, signals.csv, run_manifest.json.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

import ev as E
import metrics as M
import train as T
import costs as C

EVAL_VERSION = "v1.0.0"
FOLDS = [("F1", "2026-07-06", "2026-07-26", "2026-07-27", "2026-08-03"),
         ("F2", "2026-07-06", "2026-08-02", "2026-08-03", "2026-08-10"),
         ("F3", "2026-07-06", "2026-08-09", "2026-08-10", "2026-08-17"),
         ("F4", "2026-07-06", "2026-08-16", "2026-08-17", "2026-08-24"),
         ("F5", "2026-07-06", "2026-08-23", "2026-08-24", "2026-08-31"),
         ("F6", "2026-07-06", "2026-08-31", "2026-09-01", "2026-09-08")]
MODELS = ["baseline", "linear", "lgbm_A", "lgbm_B", "lgbm_C"]


def _ep(s):
    return float(np.datetime64(s).astype("datetime64[s]").astype(np.float64))


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path):
    df = pd.read_csv(path, dtype={"market_id": str})
    df["end3"] = pd.to_datetime(df["end3"], utc=True)
    return df


def main(indir=None, outdir=None):
    import build_dataset as B
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("MKTREL_ROOT", os.path.dirname(base))
    indir = indir or os.path.join(root, "build")
    outdir = outdir or os.path.join(root, "eval")
    os.makedirs(outdir, exist_ok=True)

    df = load_dataset(os.path.join(indir, "dataset.csv"))
    feats = B.FEATURES
    Xall = df[feats].to_numpy(dtype=np.float64)
    yall = df["label"].to_numpy(dtype=np.float64)
    pall = df["p_market_yes"].to_numpy(dtype=np.float64)
    ends = np.asarray(B.epoch_of(df["end3"]), dtype=np.float64)
    mids = df["market_id"].to_numpy()
    yas = df["yes_ask"].to_numpy(dtype=np.float64)
    nas = df["no_ask"].to_numpy(dtype=np.float64)
    assets = df["asset"].to_numpy()
    j_by_mid = {m: j for j, m in enumerate(mids)}

    oof = pd.DataFrame({"market_id": mids, "fold": df["fold"].to_numpy(),
                        "y": yall, "p_market": pall})
    for m in MODELS:
        oof[m] = np.nan
    fold_metrics = {}
    signal_rows = []
    medians = {}

    for name, tr_lo, tr_hi, va_lo, va_hi in FOLDS:
        tr_lo, tr_hi, va_lo, va_hi = _ep(tr_lo), _ep(tr_hi), _ep(va_lo), _ep(va_hi)
        tr_mask = (ends >= tr_lo) & (ends < tr_hi)
        va_mask = (ends >= va_lo) & (ends < va_hi)
        order_tr = np.argsort(ends[tr_mask], kind="mergesort")
        order_va = np.argsort(ends[va_mask], kind="mergesort")
        itr = np.nonzero(tr_mask)[0][order_tr]
        iva = np.nonzero(va_mask)[0][order_va]
        Xtr, ytr, ptr = Xall[itr], yall[itr], pall[itr]
        Xva, yva, pva = Xall[iva], yall[iva], pall[iva]
        med = np.nanmedian(Xtr, axis=0)
        medians[name] = [float(v) for v in med]
        Xtr_lin = np.where(np.isnan(Xtr), med, Xtr)
        Xva_lin = np.where(np.isnan(Xva), med, Xva)

        preds = {"baseline": T.baseline_predict(pva)}
        lin = T.train_linear(Xtr_lin, ytr, ptr, feats)
        preds["linear"] = T.linear_predict(lin, Xva_lin, pva)
        for cfg in ("A", "B", "C"):
            lm = T.train_lgbm(Xtr, ytr, ptr, cfg, feats)
            pr, _ = T.lgbm_predict(lm, Xva, pva)
            preds["lgbm_" + cfg] = pr
        for m in MODELS:
            oof.loc[oof["market_id"].isin(mids[iva]), m] = preds[m]
        fold_metrics[name] = {"n_train": int(len(itr)), "n_val": int(len(iva))}
        for m in MODELS:
            it, sl = M.cal_slope_intercept(yva, preds[m])
            fold_metrics[name][m] = {
                "logloss": M.logloss(yva, preds[m]),
                "brier": M.brier(yva, preds[m]),
                "ece": M.ece(yva, preds[m]),
                "cal": {"intercept": it, "slope": sl},
            }

    oof_path = os.path.join(outdir, "oof.csv")
    oof.to_csv(oof_path, index=False, float_format="%.10f", lineterminator="\n")

    bins = {}
    for m in MODELS:
        sub = oof.dropna(subset=[m])
        edges = E.build_bins(sub[m].to_numpy(), sub["y"].to_numpy())
        table = E.bin_table(sub[m].to_numpy(), sub["y"].to_numpy(), edges)
        bins[m] = {"edges": [float(e) for e in edges], "table": table}

    sig_all = []
    for m in MODELS:
        edges = np.array(bins[m]["edges"])
        table = bins[m]["table"]
        pm = oof.dropna(subset=[m]).reset_index(drop=True)
        for mid, pv, fold in zip(pm["market_id"].to_numpy(), pm[m].to_numpy(), pm["fold"].to_numpy()):
            j = j_by_mid[mid]
            d = E.decide_row(float(pv), float(yas[j]), float(nas[j]), edges, table)
            sig_all.append((m, mid, fold, d))
    for m, mid, fold, d in sig_all:
        if d["side"] is None:
            continue
        j = j_by_mid[mid]
        won = bool(yall[j] == 1.0)
        e = E.realized_hist(d["side"], float(pall[j]), float(yas[j]), float(nas[j]), won)
        signal_rows.append({"model": m, "market_id": mid, "fold": fold, "asset": assets[j],
                            "side": d["side"], "reason": d["reason"],
                            "p_final": d["p_final"], "point": d["point_yes"] if d["side"] == "YES" else d["point_no"],
                            "lower": d["lower_yes"] if d["side"] == "YES" else d["lower_no"],
                            "bin": d["bin"], "won": won,
                            "raw": e["raw"], "executable": e["executable"],
                            "fee07": e["fee07"], "slip_extra": e["slip_extra"],
                            "canon_net": e["canon_net"], "end3": str(df.iloc[j]["end3"])})
    sig = pd.DataFrame(signal_rows).sort_values(["model", "end3", "market_id"], kind="mergesort").reset_index(drop=True)
    sig_path = os.path.join(outdir, "signals.csv")
    sig.to_csv(sig_path, index=False, float_format="%.10f", lineterminator="\n")

    econ = {}
    for m in MODELS:
        s = sig[sig["model"] == m].sort_values(["end3", "market_id"], kind="mergesort")
        n = len(s)
        if n == 0:
            econ[m] = {"n_signals": 0}
            continue
        canon = s["canon_net"].to_numpy(dtype=np.float64)
        raw = s["raw"].to_numpy(dtype=np.float64)
        days = s["end3"].str.slice(0, 10).to_numpy()
        ci = M.block_bootstrap_ci(canon, days)
        by_fold = s.groupby("fold")["canon_net"].sum().to_dict()
        n_by_fold = s.groupby("fold").size().to_dict()
        by_asset = s.groupby("asset")["canon_net"].sum().to_dict()
        fold_share, fold_which = M.positive_share(by_fold)
        asset_share, asset_which = M.positive_share(by_asset)
        per_day = s.groupby(s["end3"].str.slice(0, 10))["canon_net"].sum()
        eq = np.cumsum(canon)
        econ[m] = {"n_signals": int(n),
                   "raw_total": float(raw.sum()),
                   "canon_total": float(canon.sum()),
                   "roi": float(canon.sum() / n),
                   "coverage": float(n / len(oof.dropna(subset=[m]))),
                   "maxdd": M.max_drawdown(eq),
                   "maxdd_over_staked": float(M.max_drawdown(eq) / n),
                   "ci95": [ci[0], ci[1]],
                   "by_fold": {k: float(v) for k, v in by_fold.items()},
                   "n_by_fold": {k: int(v) for k, v in n_by_fold.items()},
                   "fold_pos_share": [fold_share, fold_which],
                   "by_asset": {k: float(v) for k, v in by_asset.items()},
                   "asset_pos_share": [asset_share, asset_which],
                   "max_day_abs": float(per_day.abs().max()),
                   "n_days": int(len(per_day))}
    with open(os.path.join(outdir, "bins.json"), "w", newline="") as f:
        json.dump(bins, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(outdir, "metrics.json"), "w", newline="") as f:
        json.dump({"folds": fold_metrics, "econ": econ}, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(outdir, "medians.json"), "w", newline="") as f:
        json.dump(medians, f, indent=2, sort_keys=True)
        f.write("\n")
    manifest = {"eval_version": EVAL_VERSION, "models": MODELS,
                "dataset_sha256": sha_of(os.path.join(indir, "dataset.csv")),
                "config_registry_sha256": hashlib.sha256(
                    json.dumps(T.config_registry(), sort_keys=True).encode()).hexdigest(),
                "code_versions": {"train": T.MODEL_CODE_VERSION, "ev": E.EV_CODE_VERSION,
                                  "metrics": M.METRICS_VERSION, "costs": C.COST_VERSION},
                "oof_sha256": sha_of(oof_path), "signals_sha256": sha_of(sig_path)}
    with open(os.path.join(outdir, "run_manifest.json"), "w", newline="") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print("folds_done=6 signals_total=%d" % len(sig))
    for m in MODELS:
        e = econ.get(m, {})
        print("%s n=%s canon=%s roi=%s" % (m, e.get("n_signals"), round(e.get("canon_total", 0) or 0, 2), round(e.get("roi", 0) or 0, 4)))
    return manifest


if __name__ == "__main__":
    main()
