"""f5_univariate.py — F5 univariate feature screen (spec lgbm_feature_audit_v1.yaml).

Per catalog feature, train a univariate L2 logistic regression (C=1.0) on
StandardScaler-fitted train folds only (train_k_to_validation_k expanding
window, never across periods), predict the validation fold.
  - legacy target cohort: flip_native  (native_target)
  - canonical target cohort: contract_target
Per feature x target x fold: AUC + logloss. Confirmatory cohort (contract_target):
permutation p-value (labels shuffled within day x asset x regime strata on the
F5 fold, capped sample) then Holm-Bonferroni correction across all features.

Output: out/feature/UNIVARIATE_OOF.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATRIX = os.path.join(ROOT, "out", "feature", "FEATURE_MATRIX.parquet")
OUT = os.path.join(ROOT, "out", "feature", "UNIVARIATE_OOF.csv")
PERM_FOLD = "F5"
N_PERM = 200
PERM_CAP_TRAIN = 150000
PERM_CAP_VAL = 25000
SEED = 20260916

FOLD_CHAIN = ["trainpool", "F1", "F2", "F3", "F4", "F5", "F6"]
EVAL_FOLDS = ["F2", "F3", "F4", "F5", "F6"]
CATALOG = [
    "alternation_rate_6", "bb_position", "bb_width", "body_to_range",
    "consec_balance", "consec_down", "consec_up", "consecutive_down", "consecutive_up",
    "cvd_1", "cvd_6", "cvd_trend", "day_of_week",
    "dev_sq_x_phase", "deviation_x_time",
    "direction_lag_1", "direction_lag_2", "direction_lag_3",
    "dist_to_high_24", "dist_to_high_96", "dist_to_low_24", "dist_to_low_96",
    "dow", "dow_cos", "dow_sin",
    "ema_ratio_9_21",
    "funding_extreme", "funding_rate", "funding_rate_ma3",
    "high_price_final", "hour_cos", "hour_of_day", "hour_sin", "hour_utc",
    "is_final_phase", "log_moneyness", "log_time_left",
    "mid_price",
    "pm_best_ask", "pm_best_bid", "pm_momentum_5m", "pm_quote_pressure",
    "pm_spread_pct", "pm_volume_5m",
    "price_deviation", "price_deviation_sq", "price_distance_from_max",
    "price_momentum", "price_velocity", "price_velocity_lag1",
    "range_1", "range_avg_24",
    "ret_1", "ret_12", "ret_24", "ret_3", "ret_48", "ret_6",
    "rsi_14",
    "signed_body_pct", "signed_trend_efficiency_6",
    "spread", "spread_pct", "spread_trend",
    "strike_gap_pct", "taker_buy_ratio",
    "time_left_min", "time_phase",
    "up_ratio_4", "velocity_x_phase",
    "vol_24", "vol_48", "vol_6", "vol_ratio", "vol_trend", "vol_z_1", "vol_z_6",
    "volume_5min", "volume_trend",
]
TARGETS = {"native": "flip_native", "contract": "contract_target"}


def train_index_for(target, chain):
    i = chain.index(target)
    return chain[:i]


def fit_predict_eval(x_tr, y_tr, x_va, y_va):
    """L2 LogReg C=1 on standardized train, returns auc + logloss on val."""
    if len(y_tr) < 50 or len(np.unique(y_tr)) < 2 or len(y_va) < 20:
        return np.nan, np.nan
    sc = StandardScaler().fit(x_tr.reshape(-1, 1))
    xt = sc.transform(x_tr.reshape(-1, 1)).ravel()
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(xt.reshape(-1, 1), y_tr)
    xv = sc.transform(x_va.reshape(-1, 1)).ravel()
    p = np.clip(clf.predict_proba(xv.reshape(-1, 1))[:, 1], 1e-9, 1 - 1e-9)
    try:
        auc = float(roc_auc_score(y_va, p))
    except ValueError:
        auc = np.nan
    ll = float(-(y_va * np.log(p) + (1 - y_va) * np.log(1 - p)).mean())
    return auc, ll


def main():
    m = pd.read_parquet(MATRIX, columns=["fold", "decision_at", "asset"] + CATALOG + ["flip_native", "contract_target"])
    rows = []

    for fold in EVAL_FOLDS:
        tr_mask = m["fold"].isin(train_index_for(fold, FOLD_CHAIN))
        va_mask = m["fold"] == fold
        for tname, tcol in TARGETS.items():
            y_tr = m.loc[tr_mask, tcol].to_numpy(float)
            y_va = m.loc[va_mask, tcol].to_numpy(float)
            for f in CATALOG:
                x_tr = m.loc[tr_mask, f].to_numpy(float)
                x_va = m.loc[va_mask, f].to_numpy(float)
                ok_tr = np.isfinite(x_tr) & np.isfinite(y_tr)
                ok_va = np.isfinite(x_va) & np.isfinite(y_va)
                if ok_va.sum() == 0:
                    rows.append({"feature": f, "target_cohort": tname, "fold": fold,
                                 "n_train": int(ok_tr.sum()), "n_val": 0,
                                 "auc": np.nan, "logloss": np.nan,
                                 "p_perm": np.nan, "p_holm": np.nan})
                    continue
                auc, ll = fit_predict_eval(x_tr[ok_tr], y_tr[ok_tr], x_va[ok_va], y_va[ok_va])
                rows.append({"feature": f, "target_cohort": tname, "fold": fold,
                             "n_train": int(ok_tr.sum()), "n_val": int(ok_va.sum()),
                             "auc": auc, "logloss": ll,
                             "p_perm": np.nan, "p_holm": np.nan})
        print(f"fold {fold} done", flush=True)

    # ---- confirmatory permutation p on a single late validation fold ----
    rng = np.random.default_rng(SEED)
    tr_mask = m["fold"].isin(train_index_for(PERM_FOLD, FOLD_CHAIN))
    va_mask = m["fold"] == PERM_FOLD
    y_va_all = m.loc[va_mask, "contract_target"].to_numpy(float)
    date_va = m.loc[va_mask, "decision_at"].dt.date.astype(str).to_numpy()
    asset_va = m.loc[va_mask, "asset"].to_numpy()

    pvals = {}
    for f in CATALOG:
        x_tr = m.loc[tr_mask, f].to_numpy(float)
        y_tr = m.loc[tr_mask, "contract_target"].to_numpy(float)
        x_va = m.loc[va_mask, f].to_numpy(float)
        ok_tr = np.isfinite(x_tr) & np.isfinite(y_tr)
        ok_va = np.isfinite(x_va) & np.isfinite(y_va_all)
        if ok_va.sum() < 50 or ok_tr.sum() < 500:
            pvals[f] = np.nan
            continue
        xt = x_tr[ok_tr]
        yt = y_tr[ok_tr]
        if len(xt) > PERM_CAP_TRAIN:
            ix = rng.choice(len(xt), PERM_CAP_TRAIN, replace=False)
            xt, yt = xt[ix], yt[ix]
        xv = x_va[ok_va]
        yv = y_va_all[ok_va]
        d = date_va[ok_va]
        a = asset_va[ok_va]
        if len(xv) > PERM_CAP_VAL:
            ix = rng.choice(len(xv), PERM_CAP_VAL, replace=False)
            xv, yv, d, a = xv[ix], yv[ix], d[ix], a[ix]
        sta = pd.factorize(np.array([f"{a0}|{a1}" for a0, a1 in zip(d, a)]))[0]
        obs, _ = fit_predict_eval(xt, yt, xv, yv)
        if not np.isfinite(obs):
            pvals[f] = np.nan
            continue
        cnt = 1
        for _ in range(N_PERM):
            yp = yv.copy()
            for s in np.unique(sta):
                sel = np.where(sta == s)[0]
                if len(sel) < 2:
                    continue
                yp[sel] = rng.permutation(yp[sel])
            perm_auc, _ = fit_predict_eval(xt, yt, xv, yp)
            if np.isfinite(perm_auc) and perm_auc >= obs:
                cnt += 1
        pvals[f] = cnt / (N_PERM + 1)
        print(f"perm {f}: obs_auc={obs:.4f} p={pvals[f]:.4f}", flush=True)

    # Holm-Bonferroni over confirmatory cohort
    holm = {f: np.nan for f in CATALOG}
    feats = [f for f in CATALOG if pvals.get(f, np.nan) == pvals.get(f)]
    order = sorted(feats, key=lambda f: pvals[f])
    k = len(order)
    for i, f in enumerate(order):
        holm[f] = min(1.0, pvals[f] * (k - i))

    for r in rows:
        if r["target_cohort"] == "contract" and r["fold"] == PERM_FOLD:
            r["p_perm"] = pvals.get(r["feature"], np.nan)
            r["p_holm"] = holm.get(r["feature"], np.nan)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("wrote", OUT)

    sig = out[(out["target_cohort"] == "contract") & (out["fold"] == PERM_FOLD) &
              (out["p_holm"].notna())].drop_duplicates("feature")
    sig = sig.sort_values("p_holm")
    print("\nconfirmatory Holm-significant (p_holm<0.05):")
    print(sig[sig["p_holm"] < 0.05][["feature", "auc", "p_perm", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()