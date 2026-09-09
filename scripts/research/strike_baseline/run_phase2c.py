"""Phase 2c review diagnostics (protocol v1.4, frozen; NO model selection).

(a) Market control: yes_mid as probability on the same ok rows vs the SAME
    proxy outcome (diagnostic only: price refers to the real contract).
(b) Paired daily-block bootstrap CIs (B=5000, seed=0) for test Brier
    differences (model minus M1) on UTC decision days.
(c) Econ breakdown of the frozen rule by entry x side; NO leg is flagged
    SYNTHETIC (ask_no = 1 - yes_bid, never observed).
(d) Outcome-flip sensitivity: alt strike = OPEN of first 1m candle in window;
    flip rate of settlement-vs-strike labels. Task-definition sensitivity.

Model specs/params are imported from run_phase2a/run_phase2b (single source);
training is deterministic, test never fit or stopped on.
Outputs: artifacts/research/strike_baseline/sb2c_<ts>/{phase2c_metrics.json,
run_manifest.json}. Text LF-only.
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
from polyflip.research.strike_baseline.market_rules import build_window

N_BOOT = 5000
BOOT_SEED = 0


def _load(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paired_brier_ci(y: np.ndarray, p_ref: np.ndarray, p_cmp: np.ndarray,
                    days: np.ndarray, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> dict:
    """Bootstrap over day-blocks (paired rows): CI for mean Brier(cmp)-Brier(ref)."""
    uniq = np.unique(days)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    b0 = (p_ref - y) ** 2
    b1 = (p_cmp - y) ** 2
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(days, pick)
        diffs[b] = float(b1[m].mean() - b0[m].mean())
    return {"mean_diff": round(float(diffs.mean()), 6),
            "ci_lo": round(float(np.percentile(diffs, 2.5)), 6),
            "ci_hi": round(float(np.percentile(diffs, 97.5)), 6),
            "n_days": int(len(uniq)), "n_boot": n_boot, "seed": seed}


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("usage: run_phase2c.py <worktree_root> [ledger_run] [splits_run]", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).resolve()
    sb = root / "artifacts" / "research" / "strike_baseline"
    p1 = _load("run_phase1")
    p2a = _load("run_phase2a")
    p2b = _load("run_phase2b")
    ledger_run = sys.argv[2] if len(sys.argv) >= 3 else p2b.LEDGER_RUN
    splits_run = sys.argv[3] if len(sys.argv) == 4 else p2b.SPLITS_RUN

    ledger = pd.read_parquet(sb / ledger_run / "ledger.parquet")
    ok = ledger[ledger["status"] == "ok"].copy()
    ok["y"] = (ok["outcome"] == "YES").astype(int)
    splits = pd.read_csv(sb / splits_run / "splits.csv")
    ok = ok.merge(splits[["market_id", "split"]], on="market_id", how="left")
    assert ok["split"].notna().all()

    # ---- features + probs (same frozen specs as 2a/2b) ----
    raw_snaps = (sb / "data_export" / "20260909_115314" / "raw" / "snapshots.csv.gz")
    snaps = pd.read_csv(raw_snaps, compression="gzip",
                        usecols=["market_id", "recorded_at", "best_bid", "best_ask"])
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"].to_numpy(dtype="object"),
                                          utc=True, format="ISO8601")
    snaps_by_market = {mid: g for mid, g in snaps.groupby("market_id", sort=False)}
    export_dir = sb / "data_export" / "20260909_115314"
    store = p1.load_1m(export_dir)
    feat = p2b.build_features(ok, snaps_by_market, store)
    feat = p2b.add_state_dummies(p2b.add_state_dummies(feat, "ct_state", "ct"), "cs_state", "cs")

    tr = feat[feat["split"] == "train"]
    va = feat[feat["split"] == "validation"]

    def fit_logit(cols: list[str]):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000, random_state=0))
        pipe.fit(tr[cols].to_numpy(dtype=float), tr["y"].to_numpy())
        return pipe.predict_proba(feat[cols].to_numpy(dtype=float))[:, 1]

    probs = {"M0": feat["p0"].to_numpy(dtype=float),
             "M1": fit_logit(p2a.FEATS),
             "C": np.full(len(feat), float(tr["y"].mean())),
             "M2": fit_logit(p2a.FEATS + p2b.CT_D),
             "M3": fit_logit(p2a.FEATS + p2b.CS_D),
             "M4": fit_logit(p2a.FEATS + p2b.CT_D + p2b.CS_D)}
    m5cols = p2a.FEATS + p2b.CT_D + p2b.CS_D + p2b.GBM_EXTRA
    dtr = lgb.Dataset(tr[m5cols], label=tr["y"].to_numpy())
    dva = lgb.Dataset(va[m5cols], label=va["y"].to_numpy(), reference=dtr)
    booster = lgb.train(p2b.GBM_PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    probs["M5"] = np.asarray(booster.predict(feat[m5cols]), dtype=float)

    out: dict = {"protocol_version": "1.4", "input_ledger": ledger_run,
                 "input_splits": splits_run}

    # ---- (a) market control ----
    mc = {}
    for s in ("train", "validation", "test"):
        m = (feat["split"] == s).to_numpy()
        y = feat.loc[m, "y"].to_numpy(dtype=float)
        p = feat.loc[m, "yes_mid"].to_numpy(dtype=float)
        rel = p2a.reliability(y, p)
        mc[s] = {"n": int(m.sum()), "brier": round(p2a.brier(y, p), 6),
                 "log_loss": round(p2a.logloss(y, p), 6),
                 "ece_10": round(p2a.ece_from_table(rel, int(m.sum())), 6),
                 "note": "diagnostic: price of the real contract vs proxy outcome"}
    out["market_control_yes_mid"] = mc

    # ---- (b) paired daily-block CIs on test ----
    te = feat["split"] == "test"
    yte = feat.loc[te, "y"].to_numpy(dtype=float)
    dayste = feat.loc[te, "decision_at"].dt.tz_convert("UTC").dt.date.to_numpy()
    ci = {}
    for name in ("M0", "C", "M2", "M3", "M4", "M5"):
        ci[name] = paired_brier_ci(yte, np.asarray(probs["M1"])[te.to_numpy()],
                                   np.asarray(probs[name])[te.to_numpy()], dayste)
    out["paired_brier_ci_vs_M1_test"] = ci

    # ---- (c) econ by entry x side, TEST split only; NO leg is SYNTHETIC ----
    econ_split = {}
    te_idx = (feat["split"] == "test").to_numpy()
    for name in ("M0", "M1"):
        p = np.asarray(probs[name])[te_idx]
        d = feat[feat["split"] == "test"].copy()
        d["p"] = p
        d["side"] = np.where(d["p"] >= p2a.P_YES, "YES", np.where(d["p"] <= p2a.P_NO, "NO", "PASS"))
        ent = d[d["side"] != "PASS"].copy()
        yy = (ent["outcome"] == "YES").to_numpy(dtype=float)
        is_yes = (ent["side"] == "YES").to_numpy()
        ent = ent.assign(pnl=np.where(
            is_yes, (p2a.STAKE / ent["yes_ask"].to_numpy()) * yy - p2a.STAKE,
            (p2a.STAKE / (1.0 - ent["yes_bid"].to_numpy())) * (1.0 - yy) - p2a.STAKE))
        grp = {}
        for (em, side), g in ent.groupby(["entry_min_before_close", "side"]):
            grp[f"entry_{em}_side_{side}"] = {
                "n": int(len(g)), "gross": round(float(g["pnl"].sum()), 4),
                "expectancy": round(float(g["pnl"].mean()), 6),
                "synthetic_leg": side == "NO",
                "note": "NO ask = 1 - yes_bid is synthetic, never observed" if side == "NO" else "YES at observed ask"}
        econ_split[name] = grp
    out["econ_by_entry_side_test_M1_M0"] = econ_split
    out["econ_split_note"] = ("TEST split only; frozen rule p>=0.60 YES at observed ask / "
                              "p<=0.40 NO at SYNTHETIC ask_no = 1 - yes_bid; stake 10; gross only")

    # ---- (d) outcome-flip sensitivity (first-OPEN strike variant) ----
    mk = pd.read_csv(export_dir / "raw" / "markets.csv.gz", compression="gzip", low_memory=False)
    mk["market_id"] = mk["market_id"].astype(str)
    feat_ids = set(feat["market_id"].astype(str).unique())
    mk = mk[mk["market_id"].isin(feat_ids)]
    flips, tot = 0, 0
    for _, mrow in mk.iterrows():
        w = build_window(mrow["market_id"], mrow["asset"], mrow["question"],
                         pd.to_datetime(mrow["end_time_est"], utc=True))
        if not w.rules_ok:
            continue
        sym = ASSET_TO_SYMBOL.get(mrow["asset"], mrow["asset"])
        sc = store.first_close_in_window(
            sym, pd.Timestamp(w.window_start_utc).tz_localize("UTC"),
            pd.Timestamp(w.window_end_utc).tz_localize("UTC"))
        if sc is None:
            continue
        alt = float(sc["open"])
        sub = feat[feat["market_id"].astype(str) == str(mrow["market_id"])]
        for _, r in sub.iterrows():
            alt_out = "YES" if float(r["settlement_price"]) > alt else "NO"
            tot += 1
            flips += int(alt_out != r["outcome"])
    assert tot > 0, "no rows for outcome-flip sensitivity"
    out["outcome_flip_first_open"] = {
        "n": int(tot), "flips": int(flips),
        "flip_rate": round(float(flips / tot), 6) if tot else None,
        "note": "settlement vs first-candle OPEN instead of first-candle CLOSE; task-definition sensitivity"}

    run_id = f"sb2c_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    outdir = sb / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    p2a.write_text_lf(outdir / "phase2c_metrics.json",
                      json.dumps(out, indent=2, sort_keys=True) + "\n")
    payload = {"run_id": run_id, "phase": "2c",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "commit": p2a._git_commit(root),
               "protocol_sha256": p2a.sha256_file(root / "research" / "strike_baseline" / "protocol.yaml"),
               "input_ledger": ledger_run,
               "ledger_sha256": p2a.sha256_file(sb / ledger_run / "ledger.parquet"),
               "input_splits": splits_run,
               "splits_sha256": p2a.sha256_file(sb / splits_run / "splits.csv"),
               "artifacts": {}}
    for pth in sorted(outdir.iterdir()):
        if pth.is_file():
            payload["artifacts"][pth.name] = {"sha256": p2a.sha256_file(pth), "bytes": pth.stat().st_size}
    p2a.write_manifest(outdir / "run_manifest.json", payload)

    print(f"\nRUN_COMPLETE dir={outdir}")
    print("market_control test brier:", mc["test"]["brier"])
    for name, r in ci.items():
        print(f"  dBrier({name}-M1): mean={r['mean_diff']} 95%CI=[{r['ci_lo']},{r['ci_hi']}] days={r['n_days']}")
    print("flip_rate:", out["outcome_flip_first_open"]["flip_rate"])


if __name__ == "__main__":
    main()
