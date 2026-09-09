"""Phase 2b for strike-baseline research (protocol v1.3, frozen).

CT/CS causal features + M2/M3/M4 (frozen logistic) + M5 (frozen LightGBM).
Splits REUSED from sb2a run (identical markets; test never fit/stopped on).
M5: train=train, early_stopping(100, max 1000) on VALIDATION only, report test.

Metric/economic harness reused from run_phase2a (single source, importlib).
Outputs: artifacts/research/strike_baseline/sb2b_<ts>/{phase2b_metrics.json,
reliability.csv, feature_coverage.json, run_manifest.json}. Text LF-only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from polyflip.research.strike_baseline.dataset import ASSET_TO_SYMBOL
from polyflip.research.strike_baseline.regime import (
    add_state_dummies, cs_for_decision, ct_for_decision,
)

LEDGER_RUN = "sb_20260909_131059"
SPLITS_RUN = "sb2a_20260909_131711"
# v1.4: realized horizon (wend - snapshot_at), not the entry-grid label.
BASE_FEATS = ["z", "time_left_min", "sigma_min"]
CT_D = ["ct_reversion", "ct_trend", "ct_quiet"]
CS_D = ["cs_reversion", "cs_trend", "cs_quiet"]
GBM_EXTRA = ["ct_er", "cs_er"]
GBM_PARAMS = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05,
              "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
              "random_state": 0, "deterministic": True, "verbose": -1}


def _load(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _naive64(at: pd.Timestamp) -> np.datetime64:
    if at.tzinfo is not None:
        at = at.tz_convert("UTC").tz_localize(None)
    return at.to_datetime64()


def build_features(ok: pd.DataFrame, snaps_by_market: dict, store) -> pd.DataFrame:
    """Causal CT/CS per ok row. Past-only inputs (tested in synthetic_checks)."""
    ct_states, ct_ers, ct_ns = [], [], []
    cs_states, cs_ers = [], []
    # Per-market sorted snapshot arrays (once).
    prepared = {}
    for mid, g in snaps_by_market.items():
        g = g.sort_values("recorded_at")
        t = pd.to_datetime(g["recorded_at"], utc=True)
        prepared[mid] = (t.to_numpy(dtype="datetime64[ns]").astype("int64"),
                         g["best_bid"].to_numpy(dtype=float),
                         g["best_ask"].to_numpy(dtype=float),
                         g)
    n_total = len(ok)
    for i, (_, r) in enumerate(ok.iterrows()):
        if i and i % 10000 == 0:
            print(f"[2b] features {i}/{n_total} ...", flush=True)
        mid = r["market_id"]
        sat = r["snapshot_at"]
        # --- CT: prior mids strictly before snapshot ---
        if mid in prepared:
            tns, bids, asks, g = prepared[mid]
            cut = _naive64(pd.Timestamp(sat)).astype("datetime64[ns]").astype("int64")
            idx = int(np.searchsorted(tns, cut, side="left"))
            lo = max(0, idx - 30)
            sub = g.iloc[lo:idx]
        else:
            sub = None
        if sub is None or sub.empty:
            ctr = {"ct_state": "UNCERTAIN", "ct_er": np.nan, "ct_n_mids": 0}
        else:
            ctr = ct_for_decision(sub, pd.Timestamp(sat))
        ct_states.append(ctr["ct_state"])
        ct_ers.append(ctr["ct_er"])
        ct_ns.append(ctr.get("ct_n_mids", 0))
        # --- CS: last <=30 closed 1m candles (close_time <= snapshot) ---
        sym = ASSET_TO_SYMBOL.get(r["asset"], r["asset"])
        g = store.by_symbol.get(sym)
        if g is None:
            csr = {"cs_state": "UNCERTAIN", "cs_er": np.nan}
        else:
            ct_arr = store.close_ns[sym]
            hi = int(np.searchsorted(ct_arr, _naive64(pd.Timestamp(sat)), side="right"))
            lo = max(0, hi - 30)
            closes = g["close"].to_numpy(dtype=float)[lo:hi]
            times = ct_arr[lo:hi]
            csr = cs_for_decision(closes, times, pd.Timestamp(sat))
        cs_states.append(csr["cs_state"])
        cs_ers.append(csr["cs_er"])
    out = ok.copy()
    out["ct_state"] = ct_states
    out["ct_er"] = ct_ers
    out["ct_n_mids"] = ct_ns
    out["cs_state"] = cs_states
    out["cs_er"] = cs_ers
    return out


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("usage: run_phase2b.py <worktree_root> [ledger_run] [splits_run]", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).resolve()
    ledger_run = sys.argv[2] if len(sys.argv) >= 3 else LEDGER_RUN
    splits_run = sys.argv[3] if len(sys.argv) == 4 else SPLITS_RUN
    sb = root / "artifacts" / "research" / "strike_baseline"
    p1 = _load("run_phase1")
    p2a = _load("run_phase2a")

    ledger = pd.read_parquet(sb / ledger_run / "ledger.parquet")
    ok = ledger[ledger["status"] == "ok"].copy()
    ok["y"] = (ok["outcome"] == "YES").astype(int)

    splits = pd.read_csv(sb / splits_run / "splits.csv")
    ok = ok.merge(splits[["market_id", "split"]], on="market_id", how="left")
    assert ok["split"].notna().all(), "2b markets differ from 2a splits"
    assert ok.groupby("market_id")["split"].nunique().max() == 1

    raw_snaps = (Path(__file__).resolve().parents[3] / "artifacts" / "research" /
                 "strike_baseline" / "data_export" / "20260909_115314" / "raw" / "snapshots.csv.gz")
    snaps = pd.read_csv(raw_snaps, compression="gzip",
                        usecols=["market_id", "recorded_at", "best_bid", "best_ask"])
    # NOTE: pandas3 Arrow-string Series breaks to_datetime's cache path on 8M
    # rows; go through an object ndarray (identical values, parses as ISO8601).
    snaps["recorded_at"] = pd.to_datetime(
        snaps["recorded_at"].to_numpy(dtype="object"), utc=True, format="ISO8601")
    snaps_by_market = {mid: g for mid, g in snaps.groupby("market_id", sort=False)}

    export_dir = (Path(__file__).resolve().parents[3] / "artifacts" / "research" /
                  "strike_baseline" / "data_export" / "20260909_115314")
    store = p1.load_1m(export_dir)

    print("[2b] building CT/CS features ...", flush=True)
    feat = build_features(ok, snaps_by_market, store)
    feat = add_state_dummies(add_state_dummies(feat, "ct_state", "ct"), "cs_state", "cs")

    tr = feat[feat["split"] == "train"]
    va = feat[feat["split"] == "validation"]
    te = feat[feat["split"] == "test"]

    def fit_logit(cols: list[str]):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000, random_state=0))
        pipe.fit(tr[cols].to_numpy(dtype=float), tr["y"].to_numpy())
        return pipe.predict_proba(feat[cols].to_numpy(dtype=float))[:, 1]

    probs = {
        "M2": fit_logit(BASE_FEATS + CT_D),
        "M3": fit_logit(BASE_FEATS + CS_D),
        "M4": fit_logit(BASE_FEATS + CT_D + CS_D),
    }
    # M5 LightGBM: train=train, early stop on validation, report test.
    m5cols = BASE_FEATS + CT_D + CS_D + GBM_EXTRA
    dtr = lgb.Dataset(tr[m5cols], label=tr["y"].to_numpy())
    dva = lgb.Dataset(va[m5cols], label=va["y"].to_numpy(), reference=dtr)
    booster = lgb.train(GBM_PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    probs["M5"] = booster.predict(feat[m5cols])
    print(f"[2b] M5 best_iteration={booster.best_iteration}", flush=True)

    metrics: dict = {"protocol_version": "1.4", "input_ledger": ledger_run,
                     "input_splits": splits_run, "n_ok": int(len(feat)),
                     "gbm_best_iteration": int(booster.best_iteration), "models": {}}
    rel_frames = []
    for mname, p in probs.items():
        mrec: dict = {"prob": {}, "econ": {}}
        for s in ("train", "validation", "test"):
            m = (feat["split"] == s).to_numpy()
            y = feat.loc[m, "y"].to_numpy(dtype=float)
            pp = np.asarray(p[m], dtype=float)
            rel = p2a.reliability(y, pp)
            rel["model"] = mname
            rel["split"] = s
            rel_frames.append(rel)
            mrec["prob"][s] = {"n": int(m.sum()), "yes_rate": round(float(y.mean()), 6),
                               "brier": round(p2a.brier(y, pp), 6),
                               "log_loss": round(p2a.logloss(y, pp), 6),
                               "ece_10": round(p2a.ece_from_table(rel, int(m.sum())), 6)}
            mrec["econ"][s] = p2a.economic(feat.loc[m].reset_index(drop=True), pp)
        metrics["models"][mname] = mrec

    coverage = {
        "ct_state": feat["ct_state"].value_counts().to_dict(),
        "cs_state": feat["cs_state"].value_counts().to_dict(),
        "ct_non_uncertain_rate": round(float((feat["ct_state"] != "UNCERTAIN").mean()), 4),
        "cs_non_uncertain_rate": round(float((feat["cs_state"] != "UNCERTAIN").mean()), 4),
        "ct_n_mids_median": float(feat["ct_n_mids"].median()),
    }

    run_id = f"sb2b_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out = sb / run_id
    out.mkdir(parents=True, exist_ok=True)
    p2a.write_text_lf(out / "phase2b_metrics.json",
                      json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    p2a.write_text_lf(out / "reliability.csv",
                      pd.concat(rel_frames, ignore_index=True).to_csv(index=False, lineterminator="\n"))
    p2a.write_text_lf(out / "feature_coverage.json",
                      json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    payload = {"run_id": run_id, "phase": "2b",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "commit": p2a._git_commit(root),
               "protocol_sha256": p2a.sha256_file(root / "research" / "strike_baseline" / "protocol.yaml"),
               "input_ledger": ledger_run,
               "ledger_sha256": p2a.sha256_file(sb / ledger_run / "ledger.parquet"),
               "input_splits": splits_run,
               "splits_sha256": p2a.sha256_file(sb / splits_run / "splits.csv"),
               "artifacts": {}}
    for pth in sorted(out.iterdir()):
        if pth.is_file():
            payload["artifacts"][pth.name] = {"sha256": p2a.sha256_file(pth), "bytes": pth.stat().st_size}
    p2a.write_manifest(out / "run_manifest.json", payload)

    print(f"\nRUN_COMPLETE dir={out}")
    for mname in ("M2", "M3", "M4", "M5"):
        t = metrics["models"][mname]["prob"]["test"]
        print(f"  {mname} test: brier={t['brier']} logloss={t['log_loss']} ece={t['ece_10']} n={t['n']}")


if __name__ == "__main__":
    main()
