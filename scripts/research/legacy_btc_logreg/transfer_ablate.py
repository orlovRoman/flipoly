"""Stage 5-6: logit audit, drop-one retrains, recipe-vs-period, transfer (LOCAL).

Train slice: operation-window opps with decision_at < 2026-08-30 (per asset).
Eval slice: common window 08-30..09-09 (same rows as the forecast tables).
Retrains use the OLD recipe exactly (LogReg C=0.1 lbfgs, flip target, no
scaler — v11 ships none) with distinct names (never overwriting v11).
BTC_OFFSET_LOCAL fits ONLY the intercept on the adapt slice (coefs frozen).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

FEATS = ["mid_price", "spread", "time_left_min"]
SPLIT = "2026-08-30T00:00:00Z"


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_intercept_only(X, y, coef) -> float:
    """Newton 1-D for intercept with frozen coefs (L2 like C=0.1 -> l2=1/(2C n))."""
    off = X @ np.asarray(coef, dtype=float)
    b, l2 = 0.0, 1.0 / (2 * 0.1 * max(len(y), 1))
    for _ in range(200):
        p = sig(off + b)
        g = float(np.mean(p - y)) + l2 * b
        h = float(np.mean(p * (1 - p))) + l2
        step = g / max(h, 1e-12)
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b)


def brier(ps, ys) -> float:
    ps = np.asarray(ps, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return float(np.mean((ps - ys) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="scored_common.pkl (operation window)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_pickle(a.scored)
    df["decision_at"] = pd.to_datetime(df["decision_at"], utc=True)
    tr = df[df["decision_at"] < SPLIT]
    ev = df[df["decision_at"] >= SPLIT]
    out: dict = {"split": SPLIT, "assets": {}}
    v11 = {"coef": [-0.20202834, -0.26294307, 0.01182981], "b": -0.75918691}
    for asset, gtr in tr.groupby("asset"):
        gev = ev[ev["asset"] == asset]
        if gev.empty or gtr.empty:
            continue
        Xtr, ytr = gtr[FEATS].to_numpy(float), gtr["target_flip"].to_numpy(int)
        Xev, yev = gev[FEATS].to_numpy(float), gev["target_flip"].to_numpy(int)
        rec: dict = {"n_train": len(gtr), "n_eval": len(gev)}
        # 24: logit contributions of v11 (mean |coef*x| share, eval slice)
        contrib = np.abs(np.asarray(v11["coef"]) * Xev).mean(axis=0)
        rec["v11_logit_share"] = {
            f: round(float(c / contrib.sum()), 4) for f, c in zip(FEATS, contrib)}
        rec["v11_brier_eval"] = round(brier(sig(Xev @ v11["coef"] + v11["b"]), yev), 5)
        # 25/26: full + drop-one retrains, OLD recipe, distinct names
        variants: dict[str, list[str]] = {
            f"{asset}_R26_FULL": FEATS,
            f"{asset}_R26_NOMID": ["spread", "time_left_min"],
            f"{asset}_R26_NOSPREAD": ["mid_price", "time_left_min"],
            f"{asset}_R26_NOTLM": ["mid_price", "spread"],
        }
        for name, feats in variants.items():
            clf = LogisticRegression(C=0.1, max_iter=2000)
            clf.fit(Xtr[:, [FEATS.index(f) for f in feats]], ytr)
            pv = sig(clf.decision_function(Xev[:, [FEATS.index(f) for f in feats]]))
            rec[name] = {"feats": feats,
                         "coef": [round(float(c), 5) for c in clf.coef_[0]],
                         "b": round(float(clf.intercept_[0]), 5),
                         "brier_eval": round(brier(pv, yev), 5)}
        # 31: BTC_FROZEN (v11 weights on this asset) is v11_brier_eval above
        # when asset != BTC; 31/34: BTC_OFFSET_LOCAL (intercept-only adapt)
        b_off = fit_intercept_only(Xtr, ytr, v11["coef"])
        pv = sig(Xev @ np.asarray(v11["coef"]) + b_off)
        rec["BTC_OFFSET_LOCAL"] = {"b": round(b_off, 5),
                                   "brier_eval": round(brier(pv, yev), 5)}
        out["assets"][asset] = rec
        print(asset, "v11:", rec["v11_brier_eval"],
              "FULL:", rec[f"{asset}_R26_FULL"]["brier_eval"],
              "OFF:", rec["BTC_OFFSET_LOCAL"]["brier_eval"])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
