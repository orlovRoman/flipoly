"""build_features.py — L0/L2 features from causal series + L1 join (spec v1.1.0).

Reads FLIP_OPPORTUNITIES.parquet (OK rows), series (snapsdec+flipslice),
FEATURE_MATRIX.parquet (79 cols via anchor_event_id).
Writes TRAIN_MATRIX.parquet to D:\\lgbm-favorite-flip-v1\\out\\.
All history strictly <= snapshot time (causal).
"""
import glob
import os

import numpy as np
import pandas as pd

AUDIT = r"D:\lgbm-audit-v1"
OUTDIR = r"D:\lgbm-favorite-flip-v1\out"
CACHE = r"D:\lgbm-favorite-flip-v1\cache"

L0_COLS = ["outsider_mid", "time_left_min", "spread", "asset",
           "favorite_side", "ret_60s", "ret_180s", "ret_300s", "vol_5m",
           "price_velocity", "volume_5min"]
# Storage names: L0 snapshot-built cols that collide with L1 matrix cols
# get l0_ prefix (same quantities as spec, disambiguated storage).
L0_RENAME = {"time_left_min": "l0_time_left_min", "spread": "l0_spread",
             "price_velocity": "l0_price_velocity",
             "volume_5min": "l0_volume_5min"}
L0_MODEL = [L0_RENAME.get(c, c) for c in L0_COLS]
L2_COLS = ["fav_max", "drop_from_max", "time_since_cross_05", "tenure_s",
           "n_flips", "recovery_180s", "fav_vel_60s", "fav_vel_180s",
           "spread_change_180s", "quote_age_s"]
META_EXCLUDE = {"decision_event_id", "market_id", "asset", "fold",
                "decision_at", "contract_target", "flip_native",
                "synthetic_no"}
LEAK_COLS = {"contract_target", "flip_native", "final_outcome",
             "p_logreg_flip", "p_market_flip", "candidate_ask", "favorite_flip"}

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_s(s):
    """Resolution-agnostic epoch seconds (float), tz-aware safe."""
    return ((pd.to_datetime(s, utc=True) - _EPOCH)
            / pd.Timedelta("1s")).to_numpy(dtype=np.float64)


def asof(times, vals, t):
    """Latest val with times <= t (times sorted asc, epoch float arrays)."""
    k = np.searchsorted(times, t, side="right") - 1
    if k < 0:
        return np.nan
    return vals[k]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    op = pd.read_parquet(os.path.join(OUTDIR, "FLIP_OPPORTUNITIES.parquet"))
    ok = op[op["reason_code"] == "OK"].copy()
    print("OK rows:", len(ok), flush=True)

    mx = pd.read_parquet(os.path.join(AUDIT, "out", "feature",
                                      "FEATURE_MATRIX.parquet"))
    feat79 = [c for c in mx.columns if c not in META_EXCLUDE]
    assert len(feat79) == 79, len(feat79)
    assert not (set(feat79) & LEAK_COLS)
    msub = mx[["decision_event_id"] + feat79].copy()
    msub["decision_event_id"] = msub["decision_event_id"].astype(np.int64)
    ok["anchor_event_id"] = ok["anchor_event_id"].astype(np.int64)
    ok = ok.merge(msub, left_on="anchor_event_id",
                  right_on="decision_event_id", how="left", validate="m:1")
    miss = ok[feat79].isna().all(axis=1).sum()
    print("rows missing all L1:", miss)
    assert miss == 0

    # series cache
    sp = os.path.join(CACHE, "series.parquet")
    if os.path.exists(sp):
        se = pd.read_parquet(sp)
    else:
        parts = []
        for fp in sorted(glob.glob(os.path.join(
                AUDIT, "data", "exp", "snapsdec", "snapsdec_*.csv.gz"))):
            s = pd.read_csv(fp, usecols=["market_id", "recorded_at",
                                         "mid_price", "spread"],
                            dtype={"market_id": str}, low_memory=False)
            parts.append(s)
        fl = pd.read_csv(os.path.join(AUDIT, "data", "exp", "flipslice",
                                      "flipslice.csv.gz"),
                         usecols=["market_id", "recorded_at", "mid_price",
                                  "spread"],
                         dtype={"market_id": str}, low_memory=False,
                         on_bad_lines="skip")
        parts.append(fl)
        se = pd.concat(parts, ignore_index=True)
        se["recorded_at"] = pd.to_datetime(se["recorded_at"], utc=True,
                                           format="mixed")
        se = se.dropna(subset=["recorded_at"]).sort_values(
            ["market_id", "recorded_at"], kind="mergesort")
        se = se.drop_duplicates(subset=["market_id", "recorded_at"],
                                keep="last")
        se.to_parquet(sp, index=False)
    print("series rows:", len(se), flush=True)
    se_by_mk = {mk: g for mk, g in se.groupby("market_id", sort=True)}

    ok["snap_recorded_at"] = pd.to_datetime(ok["snap_recorded_at"], utc=True)
    ok["market_end_at"] = pd.to_datetime(ok["market_end_at"], utc=True)
    feats = []
    for _, r in ok.iterrows():
        mk = r["market_id"]
        fav = r["favorite_side"]
        st = float(epoch_s(pd.Series([r["snap_recorded_at"]]))[0])
        g = se_by_mk.get(mk)
        d = {"n_hist_points": 0}
        if g is None:
            for c in L0_COLS + L2_COLS:
                if c not in ("outsider_mid", "time_left_min", "spread",
                             "asset", "favorite_side", "price_velocity",
                             "volume_5min", "quote_age_s"):
                    d[c] = np.nan
        else:
            rt = epoch_s(g["recorded_at"])
            h = g.iloc[np.nonzero(rt <= st + 1e-9)[0]]
            ym = pd.to_numeric(h["mid_price"], errors="coerce").to_numpy(
                dtype=np.float64)
            tt = epoch_s(h["recorded_at"])
            good = np.isfinite(ym) & (ym > 0) & (ym < 1)
            ym, tt = ym[good], tt[good]
            sp = pd.to_numeric(h["spread"], errors="coerce").to_numpy(
                dtype=np.float64)[good]
            d["n_hist_points"] = int(len(ym))
            y0 = float(r["yes_mid"])
            for X, nm in ((60, "ret_60s"), (180, "ret_180s"),
                          (300, "ret_300s")):
                yx = asof(tt, ym, st - X)
                d[nm] = (y0 / yx - 1.0) if np.isfinite(yx) and yx > 0 else np.nan
            # vol_5m: std of 60s-step returns over trailing 300s
            rets = []
            for k in range(5):
                a = asof(tt, ym, st - 60 * k)
                b = asof(tt, ym, st - 60 * (k + 1))
                if np.isfinite(a) and np.isfinite(b) and b > 0:
                    rets.append(a / b - 1.0)
            d["vol_5m"] = float(np.std(rets)) if len(rets) >= 3 else np.nan
            # L2 favorite-price history
            fp = ym if fav == "YES" else 1.0 - ym
            f0 = (y0 if fav == "YES" else 1.0 - y0)
            if len(fp) == 0:
                for c in L2_COLS:
                    d[c] = np.nan
            else:
                d["fav_max"] = float(np.max(fp))
                d["drop_from_max"] = float(np.max(fp) - f0)
                sgn = np.sign(ym - 0.5)
                chg = np.nonzero(sgn[1:] != sgn[:-1])[0]
                d["n_flips"] = int(len(chg))
                if len(chg):
                    t_cross = tt[chg[-1] + 1]
                    d["time_since_cross_05"] = float(st - t_cross)
                    d["tenure_s"] = float(st - t_cross)
                else:
                    d["time_since_cross_05"] = np.nan
                    d["tenure_s"] = float(st - tt[0])
                r180 = fp[tt >= st - 180]
                d["recovery_180s"] = (float(f0 - np.min(r180))
                                     if len(r180) else np.nan)
                for X, nm in ((60, "fav_vel_60s"), (180, "fav_vel_180s")):
                    fx = asof(tt, fp, st - X)
                    d[nm] = ((f0 - fx) / X) if np.isfinite(fx) else np.nan
                s_now = asof(tt, sp, st)
                s_old = asof(tt, sp, st - 180)
                d["spread_change_180s"] = (
                    float(s_now - s_old) if np.isfinite(s_now)
                    and np.isfinite(s_old) else np.nan)
        d["outsider_mid"] = float(r["outsider_mid"])
        d["time_left_min"] = float(
            (r["market_end_at"].value / 1e9 - st) / 60.0)
        d["spread"] = float(r["snap_spread"]) if pd.notna(
            r["snap_spread"]) else np.nan
        d["asset"] = r["asset"]
        d["favorite_side"] = fav
        d["price_velocity"] = (float(r["snap_price_velocity"])
                               if pd.notna(r["snap_price_velocity"])
                               else np.nan)
        d["volume_5min"] = (float(r["snap_volume_5min"])
                            if pd.notna(r["snap_volume_5min"]) else np.nan)
        d["quote_age_s"] = float(r["snapshot_age_seconds"])
        feats.append(d)
    F = pd.DataFrame(feats, index=ok.index)
    F = F.rename(columns=L0_RENAME)
    for c in L0_MODEL + L2_COLS:
        ok[c] = F[c].values if c in F.columns else np.nan
    ok["n_hist_points"] = F["n_hist_points"].values
    assert not (set(L0_MODEL + L2_COLS + feat79) & LEAK_COLS), "leak col in feats"
    assert len(set(L0_MODEL + L2_COLS + feat79)) == len(L0_MODEL + L2_COLS + feat79), "dup feature cols"
    import json as _json
    with open(os.path.join(OUTDIR, "FEATURE_SETS.json"), "w") as f:
        _json.dump({"L0": L0_MODEL,
                    "L1": feat79,
                    "L2": feat79 + L2_COLS,
                    "L3": L0_MODEL + L2_COLS,
                    "categorical": ["asset", "favorite_side"]}, f, indent=1)
    keep = ["opportunity_id", "market_id", "fold",
            "decision_window", "decision_at", "market_end_at",
            "candidate_side", "favorite_flip",
            "yes_mid", "outsider_mid_provenance",
            "candidate_ask", "candidate_ask_provenance",
            "p_market_flip", "p_logreg_flip", "logreg_available",
            "logreg_model_key", "logreg_model_version", "econ_available",
            "out_of_band", "flipnative_agree", "side_agree",
            "anchor_event_id", "n_hist_points",
            "snapshot_age_seconds", "anchor_age_seconds"] + L0_MODEL + \
        L2_COLS + feat79
    # note: asset/favorite_side/outsider_mid live in L0_MODEL (same values)
    tm = ok[keep].copy()
    tm.to_parquet(os.path.join(OUTDIR, "TRAIN_MATRIX.parquet"), index=False)
    print("TRAIN_MATRIX rows=%d cols=%d" % tm.shape)
    print("L0 null frac:")
    print(tm[L0_MODEL].isna().mean().round(3).to_string())
    print("L2 null frac:")
    print(tm[L2_COLS].isna().mean().round(3).to_string())
    print("flip rate by fold:")
    print(tm.groupby("fold")["favorite_flip"].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
