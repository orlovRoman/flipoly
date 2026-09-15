"""E2 native metrics (recorded probabilities, no replay needed).

Per used version: metrics vs its native target + controls, per fold.
Legacy -> legacy_native; canonical -> contract_target; flip handled via
entry path separately (not direction slots).
Controls: prevalence, always-UP, last-candle-sign (accuracy only),
momentum L2 (ret_1 last closed same interval, train-only scaler),
market-mid (canonical only).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, roc_auc_score)

ROOT = r"D:\lgbm-audit-v1"
DS = os.path.join(ROOT, "out", "shared_ds")
OUT = os.path.join(ROOT, "out", "direction")
os.makedirs(OUT, exist_ok=True)

FOLDS = ["trainpool", "F1", "F2", "F3", "F4", "F5", "F6"]
VAL_START = {"F1": "2026-07-27", "F2": "2026-08-03", "F3": "2026-08-10",
             "F4": "2026-08-17", "F5": "2026-08-24", "F6": "2026-09-01"}
SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
              "DOGE": "DOGEUSDT", "XRP": "XRPUSDT"}


def last_closed_sign(df, candles):
    """Sign of last closed 15m candle (accuracy-only control). Returns array."""
    out = np.full(len(df), np.nan)
    dec = ((df["dt"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy(dtype=float)
    for a, sym in SYMBOL_MAP.items():
        m = (df["asset"] == a).to_numpy()
        if not m.any():
            continue
        g = candles[(candles["symbol"] == sym) & (candles["interval"] == "15m")]
        g = g.sort_values("close_time").reset_index(drop=True)
        ct = ((g["close_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy(dtype=float)
        o = g["open"].to_numpy(dtype=float)
        c = g["close"].to_numpy(dtype=float)
        j = np.searchsorted(ct, dec[m], side="right") - 1
        ok = (j >= 0) & (j < len(ct))
        jj = np.clip(j, 0, len(ct) - 1)
        sgn = np.sign(c[jj] / np.where(o[jj] == 0, np.nan, o[jj]) - 1.0)
        sgn[~np.isfinite(sgn)] = np.nan
        sgn[sgn == 0] = np.nan
        vals = np.full(m.sum(), np.nan)
        vals[ok] = (sgn[ok] > 0).astype(float)
        out[m] = vals
    return out


def ret1_last_closed(df, candles):
    out = np.full(len(df), np.nan)
    dec = ((df["dt"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy(dtype=float)
    for a, sym in SYMBOL_MAP.items():
        m = (df["asset"] == a).to_numpy()
        if not m.any():
            continue
        g = candles[(candles["symbol"] == sym) & (candles["interval"] == "15m")]
        g = g.sort_values("close_time").reset_index(drop=True)
        ct = ((g["close_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy(dtype=float)
        o = g["open"].to_numpy(dtype=float)
        c = g["close"].to_numpy(dtype=float)
        j = np.searchsorted(ct, dec[m], side="right") - 1
        ok = (j >= 1) & (j < len(ct))
        jj = np.clip(j, 1, len(ct) - 1)
        r = c[jj] / np.where(o[jj] == 0, np.nan, o[jj]) - 1.0
        vals = np.full(m.sum(), np.nan)
        vals[ok] = r[ok]
        out[m] = vals
    return out


def ece_score(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return tot / len(y)


def cal_slope(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9)).reshape(-1, 1)
    try:
        m = LR().fit(lp, y)
        return float(m.intercept_[0]), float(m.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def metrics_for(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = int(len(y))
    out = {"n": m, "accuracy": float(accuracy_score(y, p > 0.5)),
           "bal_accuracy": float(balanced_accuracy_score(y, p > 0.5)),
           "brier": float(brier_score_loss(y, p)),
           "logloss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
           "ece": float(ece_score(y, p)),
           "coverage": 1.0}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = float("nan")
    it, sl = cal_slope(y, p)
    out["cal_intercept"] = it
    out["cal_slope"] = sl
    return out


def main():
    df = pd.read_parquet(os.path.join(DS, "shared_dataset.parquet"),
                         columns=["decision_event_id", "market_id", "asset",
                                  "fold", "model_slot", "slot_version",
                                  "slot_probability", "p_market_yes",
                                  "contract_target", "legacy_native",
                                  "flip_native", "decision_at"])
    inv = pd.read_csv(os.path.join(ROOT, "out", "MODEL_INVENTORY.csv"), dtype=str)
    u = pd.read_csv(os.path.join(ROOT, "out", "MODEL_USAGE_TIMELINE.csv"),
                    dtype=str, keep_default_na=False)
    used = u[u["registry_id"] != ""].copy()
    used["key"] = used["k"] + "|" + used["v"]
    inv["key"] = inv["asset"] + "|" + inv["version"]
    umap = inv.set_index("key")["target_class"].to_dict()

    def norm_ver(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(v)
        except (TypeError, ValueError):
            return "nan"
    df["key"] = (df["model_slot"].astype(str) + "|"
                 + df["slot_version"].apply(norm_ver))
    df["tclass"] = df["key"].map(umap)
    df["p_rec"] = pd.to_numeric(df["slot_probability"], errors="coerce")
    df["dt"] = pd.to_datetime(df["decision_at"], utc=True)
    candles = pd.read_csv(os.path.join(ROOT, "data", "exp", "full", "candles.csv.gz"))
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    candles["close_time"] = pd.to_datetime(candles["close_time"], utc=True)
    df["last_sign"] = last_closed_sign(df, candles)
    df["ret1"] = ret1_last_closed(df, candles)
    res = []
    for (slot, ver), g in df[df["tclass"].isin(
            ["LEGACY_BINANCE_DIRECTION",
             "CANONICAL_POLYMARKET_DIRECTION"])].groupby(["model_slot", "slot_version"]):
        tc = g["tclass"].iloc[0]
        label = "legacy_native" if tc == "LEGACY_BINANCE_DIRECTION" else "contract_target"
        gg = g[g[label].isin([0.0, 1.0]) & g["p_rec"].notna()].copy()
        if len(gg) < 50:
            continue
        y = gg[label].to_numpy()
        p = gg["p_rec"].to_numpy()
        row = {"slot": slot, "version": ver, "class": tc, "n": len(gg)}
        row["model"] = metrics_for(y, p)
        prev = float(np.mean(y))
        row["controls"] = {
            "prevalence": {"logloss": float(-(prev * np.log(max(prev, 1e-12)) + (1 - prev) * np.log(max(1 - prev, 1e-12)))), "accuracy": float(max(prev, 1 - prev))},
            "always_up": {"accuracy": float(np.mean(y))},
        }
        sg = gg["last_sign"].to_numpy(dtype=float)
        ok_s = np.isfinite(sg)
        if ok_s.sum() > 50:
            row["controls"]["last_candle_sign"] = {
                "accuracy": float(accuracy_score(y[ok_s], sg[ok_s]))}
        from sklearn.preprocessing import StandardScaler
        r1 = gg["ret1"].to_numpy(dtype=float)
        ok_r = np.isfinite(r1)
        if ok_r.sum() > 200:
            starts = gg["fold"].map(
                {f: pd.Timestamp(v, tz="UTC") for f, v in VAL_START.items()})
            is_train = (gg["fold"].to_numpy() == "trainpool") | (
                gg["dt"].to_numpy() < starts.to_numpy())
            tr = ok_r & is_train
            te = ok_r & ~is_train
            if tr.sum() > 200 and te.sum() > 50 and y[tr].std() > 0:
                sc = StandardScaler().fit(r1[tr].reshape(-1, 1))
                clf = LogisticRegression(C=1.0).fit(
                    sc.transform(r1[tr].reshape(-1, 1)), y[tr])
                pm = clf.predict_proba(sc.transform(r1[ok_r].reshape(-1, 1)))[:, 1]
                te_pos = (~is_train)[ok_r]
                row["controls"]["momentum"] = metrics_for(y[ok_r][te_pos], pm[te_pos])
        if tc == "CANONICAL_POLYMARKET_DIRECTION":
            pm = gg["p_market_yes"].to_numpy(dtype=float)
            okm = np.isfinite(pm)
            if okm.sum() > 50:
                row["controls"]["market_mid"] = metrics_for(y[okm], np.clip(pm[okm], 1e-9, 1 - 1e-9))
        by_fold = {}
        for f in FOLDS:
            gf = gg[gg["fold"] == f]
            if len(gf) >= 50:
                by_fold[f] = metrics_for(gf[label].to_numpy(), gf["p_rec"].to_numpy())
        row["by_fold"] = by_fold
        res.append(row)
        print("%s v%s %s n=%d ll=%.4f acc=%.3f auc=%.3f folds=%d" % (
            slot, ver, tc[:7], len(gg), row["model"]["logloss"],
            row["model"]["accuracy"], row["model"]["auc"], len(by_fold)))
    with open(os.path.join(OUT, "NATIVE_METRICS.json"), "w", newline="") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    print("models evaluated:", len(res))


if __name__ == "__main__":
    main()
