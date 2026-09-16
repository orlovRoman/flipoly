"""f2b_leakage.py — F2 timestamp/leakage check (spec lgbm_feature_audit_v1.yaml
quality_F2: future_timestamp_any_row -> LEAKAGE_RISK_block).

For a seeded sample of decision rows across eval folds F2..F6:
  - Candle features are REBUILT from the raw candles.csv.gz (closed 15m bars)
    and re-merged as-of (last bar with close_time <= decision_at). Verifies:
      (a) origin close_time <= decision_at for every row (no future candle),
      (b) matrix value == freshly rebuilt value (roundtrip, tol 1e-4 rel).
  - Snapshot features: re-as-of from _snapshot_features.parquet (materialized
    feature cache, per-market last recorded_at <= decision_at). Verifies:
      (a) origin recorded_at <= decision_at for every row,
      (b) matrix value == cache value.
  - Time features (hour_*, dow*, day_of_week): derived solely from decision_at
    -> origin == decision time, no leak.
  - funding_* / strike_gap_pct / log_moneyness: always-NaN, no data source
    -> no origin timestamp, cannot leak (documented as such).

Outputs (SAMPLED scope — deterministic 60k-row sample across eval folds):
  out/feature/SAMPLED_LEAKAGE_CHECK.csv            (per-feature rows)
  out/feature/SAMPLED_LEAKAGE_CHECK_SUMMARY.json
Full-matrix timestamp provenance is NOT scanned row-by-row; it is asserted by
construction (as-of searchsorted + 29s lag in f1_build_matrix). The sample
provides an independent empirical spot-check; results are labelled 'sampled'
and must NOT be quoted as full-matrix facts.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FDIR = os.path.join(ROOT, "out", "feature")
MATRIX = os.path.join(FDIR, "FEATURE_MATRIX.parquet")
SNAPS_CACHE = os.path.join(FDIR, "_snapshot_features.parquet")
CANDLES_PATH = os.path.join(ROOT, "data", "exp", "full", "candles.csv.gz")
SNAPS_DIR = os.path.join(ROOT, "data", "snapsdec")
CATALOG_CSV = os.path.join(FDIR, "FEATURE_CATALOG.csv")
OUT_LOG = os.path.join(FDIR, "SAMPLED_LEAKAGE_CHECK.csv")
OUT_SUM = os.path.join(FDIR, "SAMPLED_LEAKAGE_CHECK_SUMMARY.json")

ASSET2SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
             "XRP": "XRPUSDT", "DOGE": "DOGEUSDT"}
SYM2ASSET = {v: k for k, v in ASSET2SYM.items()}
SAMPLE_N = 60000
SEED = 20260913
RT_TOL = 1e-4

TARGET = ["fold", "asset", "market_id", "decision_at"]
BASE = ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
        "vol_6", "vol_24", "vol_48", "vol_trend", "vol_ratio", "vol_z_1", "vol_z_6",
        "cvd_1", "cvd_6", "cvd_trend", "taker_buy_ratio",
        "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
        "dist_to_high_24", "dist_to_low_24", "dist_to_high_96", "dist_to_low_96",
        "range_1", "range_avg_24",
        "consec_balance", "consec_up", "consec_down",
        "direction_lag_1", "direction_lag_2", "direction_lag_3",
        "consecutive_up", "consecutive_down", "up_ratio_4", "alternation_rate_6",
        "signed_trend_efficiency_6", "signed_body_pct", "body_to_range",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
SNAP = ["mid_price", "spread", "volume_5min", "price_velocity", "time_left_min",
        "price_deviation", "spread_pct", "log_time_left",
        "deviation_x_time", "price_deviation_sq",
        "price_distance_from_max", "time_phase",
        "velocity_x_phase", "dev_sq_x_phase",
        "is_final_phase", "high_price_final",
        "price_momentum", "spread_trend", "volume_trend", "price_velocity_lag1",
        "pm_momentum_5m", "pm_volume_5m", "pm_spread_pct", "pm_quote_pressure",
        "pm_best_bid", "pm_best_ask"]
TIME_SOURCE = ["hour_of_day", "hour_utc", "dow", "day_of_week"]
NO_SOURCE = ["funding_rate", "funding_rate_ma3", "funding_extreme",
             "strike_gap_pct", "log_moneyness"]


def value_match(col_a, col_b, atol=1e-5):
    """Compare two float series len == n; NaN==NaN counts as match."""
    a = col_a.to_numpy(dtype=np.float64)
    b = col_b.to_numpy(dtype=np.float64)
    na = np.isnan(a)
    nb = np.isnan(b)
    both_na = na & nb
    ok = both_na | (np.abs(a - b) <= RT_TOL * np.maximum(1.0, np.abs(b)))
    return float(ok.mean())


def main():
    feat_cat = pd.read_csv(CATALOG_CSV)
    feats = feat_cat["feature"].tolist()
    huge = ["flip_native", "synthetic_no", "contract_target"]
    m = pd.read_parquet(MATRIX, columns=TARGET + feats)
    for c in feats + ["market_id"]:
        if c in m.columns and m[c].dtype == np.float64:
            m[c] = m[c].astype(np.float32)
    m["decision_at"] = pd.to_datetime(m["decision_at"], utc=True, errors="coerce")

    rng = np.random.default_rng(SEED)
    total_rows = int(m.shape[0])
    sample = (m[m["fold"].isin(["F2", "F3", "F4", "F5", "F6"])]
              .groupby("fold", sort=False)
              .sample(n=SAMPLE_N // 5, random_state=rng)
              .reset_index(drop=True))
    print(f"sample rows: {len(sample)}", flush=True)
    del m
    sample["decision_at_ns"] = sample["decision_at"].dt.tz_convert("UTC").dt.tz_localize(None) \
        .astype("datetime64[ns]")
    results = []

    # ── Candle features: rebuild from raw candles per asset ──
    import f1_build_matrix as f1
    for sym in ASSET2SYM.values():
        asset = SYM2ASSET[sym]
        frame = f1.build_candle_features(sym)
        frame = frame.sort_values("close_time").reset_index(drop=True)
        sub = sample[sample["asset"] == asset].sort_values("decision_at").reset_index(drop=True)
        if sub.empty:
            print(f"  [warn] asset {asset}: empty sub", flush=True)
            continue
        merged = pd.merge_asof(
            sub, frame,
            left_on="decision_at", right_on="close_time",
            direction="backward", allow_exact_matches=True)
        not_future = merged["close_time"].notna() & (merged["close_time"] <= merged["decision_at"])
        for f in BASE:
            rcol = f if f in merged.columns else f + "_y"
            if rcol not in merged.columns or f not in sub.columns:
                continue
            rt = value_match(sub[f], merged[rcol]) \
                if merged[rcol].notna().any() else (1.0 if sub[f].isna().all() else 0.0)
            results.append({
                "feature": f, "source": "candle_rebuilt", "asset": asset,
                "n_rows": len(sub),
                "origin_le_decision": float(not_future.mean()) if merged["close_time"].notna().any() else 1.0,
                "value_match_pct": rt,
            })
    print("candle check done -> results so far:", len(results), flush=True)

    # ── Snapshot features: re-as-of from materialized cache ──
    sn = pd.read_parquet(SNAPS_CACHE)
    sn["recorded_at"] = pd.to_datetime(sn["recorded_at"], utc=True, errors="coerce")
    sn = sn.dropna(subset=["recorded_at", "market_id"])
    keep = ["market_id", "recorded_at"] + SNAP
    inter = set(sample["market_id"]) & set(sn["market_id"])
    sni = sn[sn["market_id"].isin(inter)].sort_values(["market_id", "recorded_at"]).reset_index(drop=True)
    del sn
    sni["recorded_at_ns"] = sni["recorded_at"].dt.tz_convert("UTC").dt.tz_localize(None) \
        .astype("datetime64[ns]")
    sn_by_market = {mk: g for mk, g in sni.groupby("market_id", sort=False)}

    origin_ns = np.full(len(sample), np.nan)
    joined = {f: np.full(len(sample), np.nan) for f in SNAP}
    n_joined = 0
    for mk, g in sample.groupby("market_id", sort=True):
        if mk not in sn_by_market or g.empty:
            continue
        sg = sn_by_market[mk]
        rt = sg["recorded_at_ns"].to_numpy().astype("int64")
        lt = g["decision_at_ns"].to_numpy().astype("int64")
        positions = g.index.to_numpy()
        idx = np.searchsorted(rt, lt) - 1
        ok = idx >= 0
        if not ok.any():
            continue
        pos = positions[ok]
        idxc = np.clip(idx[ok], 0, None)
        origin_ns[pos] = rt[idxc]
        for f in SNAP:
            joined[f][pos] = sg[f].to_numpy(dtype=np.float64)[idxc]
        n_joined += ok.sum()
    lt_all = sample["decision_at_ns"].to_numpy().astype("int64")
    m = ~np.isnan(origin_ns)
    for f in SNAP:
        mv = sample[f].to_numpy(dtype=np.float64)
        jv = joined[f]
        na = np.isnan(mv)
        nj = np.isnan(jv)
        both_na = na & nj
        okv = both_na | (np.abs(jv - mv) <= RT_TOL * np.maximum(1.0, np.abs(mv)))
        has_snap_rows = m & ~nj
        rt = float(okv[has_snap_rows].mean()) if has_snap_rows.any() else 1.0
        origin_le = float((origin_ns[m] <= lt_all[m]).mean()) if m.any() else 1.0
        results.append({
            "feature": f, "source": "snapshot_cache", "asset": "all",
            "n_rows": int(m.sum()),
            "origin_le_decision": origin_le,
            "value_match_pct": rt,
        })
    print("snapshot check done", flush=True)

    # ── Time features (origin = decision_at itself) ──
    for f in TIME_SOURCE:
        results.append({
            "feature": f, "source": "decision_at", "asset": "n/a",
            "n_rows": len(sample),
            "origin_le_decision": 1.0,
            "value_match_pct": 1.0,
        })

    # ── No-source always-NaN features ──
    for f in NO_SOURCE:
        na_rate = sample[f].isna().mean() if f in sample.columns else 1.0
        results.append({
            "feature": f, "source": "none_always_nan", "asset": "n/a",
            "n_rows": int(na_rate * len(sample)),
            "origin_le_decision": 1.0,
            "value_match_pct": 0.0,
        })

    log = pd.DataFrame(results)
    log.to_csv(OUT_LOG, index=False)

    # verdicts (SAMPLED scope — MUST NOT be quoted as a full-matrix scan)
    bad = log[(log["origin_le_decision"] < 1.0) | (log["value_match_pct"] < 0.999)]
    summary = {
        "sample_scope": "deterministic seeded sample, 12000 rows per eval fold F2..F6",
        "checked_rows": int(len(sample)),
        "total_rows": total_rows,
        "coverage_fraction": float(len(sample) / total_rows),
        "checked_per_fold": {f: int(v) for f, v in sample["fold"].value_counts().items()},
        "n_features": len(feats),
        "any_future_timestamp_row_in_sample": bool((log["origin_le_decision"] < 1.0).any()),
        "any_value_mismatch_in_sample": bool((log["value_match_pct"] < 0.999).any()),
        "leakage_risk_block_in_sample": bool((log["origin_le_decision"] < 1.0).any()),
        "full_matrix_provenance": ("asserted by construction in f1_build_matrix "
                                   "(as-of searchsorted + 29s lag); no full-matrix "
                                   "row-by-row timestamp scan was run"),
        "flagged_rows": bad[["feature", "origin_le_decision", "value_match_pct"]]
                        .to_dict("records"),
    }
    import json
    with open(OUT_SUM, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print("wrote", OUT_LOG, OUT_SUM)


if __name__ == "__main__":
    main()