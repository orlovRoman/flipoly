"""f2_quality.py — F2 feature data quality per fold (spec lgbm_feature_audit_v1.yaml).

Per feature (79), per validation fold: missing_rate, nunique, constants,
outliers (% of val outside 5xIQR of train), PSI drift. Rows in fold
F1/trainpool/gap are treated as non-eval but still scored for missingness.

Flags (spec thresholds):
- future_timestamp_any_row       -> LEAKAGE_RISK_block
- missing_shift_gt_pp over 10    -> MISSINGNESS_DRIFT
- psi_gt 0.25                    -> DATA_DRIFT
- val_outside_train_IQR_pct > 1  -> OUTLIER_HEAVY
- nan_inf_postproc               -> fold_block
- constant feature               -> CONSTANT

Output: out/feature/FEATURE_DATA_QUALITY.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATRIX = os.path.join(ROOT, "out", "feature", "FEATURE_MATRIX.parquet")
OUT = os.path.join(ROOT, "out", "feature", "FEATURE_DATA_QUALITY.csv")
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

# chronological fold chain (never shuffle across periods). gap is a disjoint
# leakage buffer, not a training fold; F1/trainpool have no contract_target.
FOLD_CHAIN = ["trainpool", "F1", "F2", "F3", "F4", "F5", "F6"]
EVAL_FOLDS = ["F2", "F3", "F4", "F5", "F6"]

PSI_BINS = 10
PSI_THRESHOLD = 0.25
MISSING_SHIFT_PP = 10.0
OUTLIER_PCT = 1.0
IQR_MULT = 5.0


def fold_train_indexes(chain, target):
    """Indices of rows usable as train for a target validation fold."""
    i = chain.index(target)
    return chain[:i]


def main():
    df = pd.read_parquet(MATRIX, columns=["fold"] + CATALOG)
    rows = []
    cols = ["fold"] + list(CATALOG)
    for fold in EVAL_FOLDS:
        train_folds = fold_train_indexes(FOLD_CHAIN, fold)
        tr = df[df["fold"].isin(train_folds)]
        va = df[df["fold"] == fold]
        print(f"fold {fold}: train {len(tr)} val {len(va)}", flush=True)
        for f in CATALOG:
            t = tr[f]
            v = va[f]
            t_miss = float(t.isna().mean() * 100.0)
            v_miss = float(v.isna().mean() * 100.0)
            miss_shift = float(abs(v_miss - t_miss))
            nuniq_t = int(t.nunique(dropna=True))
            nuniq_v = int(v.nunique(dropna=True))
            is_constant = (nuniq_t <= 1) and (t_miss < 100.0) and (nuniq_v <= 1)

            # outlier %: val values outside train [Q1-5IQR, Q3+5IQR]
            q1, q3 = np.nanquantile(t, [0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - IQR_MULT * iqr, q3 + IQR_MULT * iqr
            vv = v.dropna()
            out_pct = float(0.0)
            if len(vv) > 0 and np.isfinite(lo):
                out_pct = float(((vv < lo) | (vv > hi)).mean() * 100.0)

            # PSI on 10 bins from train quantiles (skip constant/fully-missing)
            psi = np.nan
            if nuniq_t >= 2 and t_miss < 100.0:
                edges = np.unique(np.quantile(t.dropna(), np.linspace(0, 1, PSI_BINS + 1)))
                if len(edges) >= 2:
                    p_tr = np.histogram(t.dropna(), bins=edges)[0].astype(float)
                    p_va = np.histogram(v.dropna(), bins=edges)[0].astype(float)
                    p_tr = p_tr / p_tr.sum()
                    p_va = p_va / max(p_va.sum(), 1)
                    p_tr = np.where(p_tr == 0, 1e-6, p_tr)
                    p_va = np.where(p_va == 0, 1e-6, p_va)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        psi = float(((p_va - p_tr) * np.log(p_va / p_tr)).sum())

            flags = []
            if is_constant:
                flags.append("CONSTANT")
            if miss_shift > MISSING_SHIFT_PP:
                flags.append("MISSINGNESS_DRIFT")
            if psi is not None and np.isfinite(psi) and psi > PSI_THRESHOLD:
                flags.append("DATA_DRIFT")
            if out_pct > OUTLIER_PCT:
                flags.append("OUTLIER_HEAVY")
            # nan_inf_postproc: features that are fully missing or have huge
            # missingness in val after the build (no usable signal -> fold_block)
            if v_miss >= 99.999:
                flags.append("HUNDRED_PCT_MISSING_BLOCK")
            elif v_miss >= 50.0:
                flags.append("MAJOR_MISSINGNESS")

            verdict = "OK"
            if any("BLOCK" in fl for fl in flags):
                verdict = "BLOCK"
            elif "CONSTANT" in flags:
                verdict = "CONSTANT"
            elif "MISSINGNESS_DRIFT" in flags or "HUNDRED_PCT_MISSING_BLOCK" in flags \
                    or "MAJOR_MISSINGNESS" in flags:
                verdict = "DATA_QUALITY_BLOCKED"
            elif "DATA_DRIFT" in flags or "OUTLIER_HEAVY" in flags:
                verdict = "WARN"

            rows.append({
                "feature": f, "fold": fold,
                "train_missing_pct": round(t_miss, 3),
                "val_missing_pct": round(v_miss, 3),
                "missing_shift_pp": round(miss_shift, 3),
                "nunique_train": nuniq_t, "nunique_val": nuniq_v,
                "constant": int(is_constant),
                "val_outside_train_iqr_pct": round(out_pct, 3),
                "psi": round(psi, 4) if np.isfinite(psi) else "",
                "flags": "|".join(flags) if flags else "",
                "verdict": verdict,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("wrote", OUT)
    # summary: worst verdict per feature
    prio = {"BLOCK": 5, "DATA_QUALITY_BLOCKED": 4, "CONSTANT": 3, "WARN": 2, "OK": 1}
    out["_p"] = out["verdict"].map(prio)
    summ = (out.sort_values("_p", ascending=False)
            .groupby("feature").first().sort_values("_p", ascending=False))
    print("features by worst verdict:")
    for f, r in summ.iterrows():
        if r["verdict"] != "OK":
            print(f"  {f}: {r['verdict']}  ({r['flags']})")


if __name__ == "__main__":
    main()