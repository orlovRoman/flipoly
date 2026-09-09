"""Phase 2a baselines for strike-baseline research (protocol v1.2, frozen).

Models (no validation/test peeking; all choices pre-registered in protocol.yaml):
  M0: p = p0 = Phi(z). No fitting.
  M1: StandardScaler (fit TRAIN only) + LogisticRegression(C=1.0, max_iter=5000,
      random_state=0) on [z, entry_min_before_close, sigma_min]. No search.
  C:  p = train YES rate. Reference only.

Splits: unique markets sorted by end_time_est, 60/20/20; variants follow market.
Metrics: brier, log_loss (eps 1e-6 clip), ECE-10 + reliability table.
Economic rule: YES if p>=0.60 at yes_ask; NO if p<=0.40 at (1-yes_bid);
stake 10 USDC; gross only (fee unknown -> net null).

Outputs: artifacts/research/strike_baseline/sb2a_<ts>/{phase2a_metrics.json,
reliability.csv, splits.csv, run_manifest.json}. All text LF-only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEDGER_RUN = "sb_20260909_131059"
EPS = 1e-6
P_YES, P_NO = 0.60, 0.40
STAKE = 10.0
N_BINS = 10
FEATS = ["z", "entry_min_before_close", "sigma_min"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def write_manifest(path: Path, payload: dict) -> dict:
    canonical = json.dumps(payload, indent=2, sort_keys=True).replace("\r\n", "\n")
    payload = dict(payload)
    payload["manifest_file_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    final = json.dumps(payload, indent=2, sort_keys=True).replace("\r\n", "\n")
    write_text_lf(path, final + "\n")
    return payload


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    pc = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        n = int(m.sum())
        rows.append({
            "bin": i, "lo": round(float(lo), 3), "hi": round(float(hi), 3), "n": n,
            "acc": round(float(y[m].mean()), 6) if n else None,
            "conf": round(float(p[m].mean()), 6) if n else None,
        })
    return pd.DataFrame(rows)


def ece_from_table(rel: pd.DataFrame, n_total: int) -> float:
    e = 0.0
    for _, r in rel.iterrows():
        if r["n"]:
            e += (r["n"] / n_total) * abs(r["acc"] - r["conf"])
    return float(e)


def economic(df: pd.DataFrame, p: np.ndarray) -> dict:
    """Frozen trading rule; gross only. df has yes_bid/yes_ask/outcome/asset/decision_at."""
    d = df.copy()
    d["p"] = p
    d["side"] = np.where(d["p"] >= P_YES, "YES", np.where(d["p"] <= P_NO, "NO", "PASS"))
    ent = d[d["side"] != "PASS"].copy()
    n_all = len(d)
    if len(ent) == 0:
        return {"n_entries": 0, "gross": 0.0, "expectancy": None,
                "pnl_per_opportunity": 0.0, "turnover": 0.0,
                "weekly": {}, "per_asset": {}, "concentration_top_decile_share": None}
    y = (ent["outcome"] == "YES").to_numpy(dtype=float)
    is_yes = (ent["side"] == "YES").to_numpy()
    ask_yes = ent["yes_ask"].to_numpy(dtype=float)
    ask_no = 1.0 - ent["yes_bid"].to_numpy(dtype=float)
    pnl = np.where(is_yes, (STAKE / ask_yes) * y - STAKE,
                   (STAKE / ask_no) * (1.0 - y) - STAKE)
    ent = ent.assign(pnl=pnl)
    gross = float(pnl.sum())
    order = np.argsort(-pnl)
    k = max(1, int(np.ceil(0.1 * len(pnl))))
    conc = float(pnl[order[:k]].sum() / gross) if gross != 0 else None
    iso = ent["decision_at"].dt.tz_convert("UTC").dt.isocalendar()
    ent = ent.assign(_yw=iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))
    weekly = {k: round(float(g["pnl"].sum()), 4) for k, g in ent.groupby("_yw")}
    per_asset = {a: {"n": int(g.shape[0]), "gross": round(float(g["pnl"].sum()), 4)}
                 for a, g in ent.groupby("asset")}
    return {"n_entries": int(len(ent)), "gross": round(gross, 4),
            "expectancy": round(float(gross / len(ent)), 6),
            "pnl_per_opportunity": round(float(gross / n_all), 6),
            "turnover": round(float(len(ent) * STAKE), 2),
            "weekly": weekly, "per_asset": per_asset,
            "concentration_top_decile_share": (round(conc, 4) if conc is not None else None)}


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: run_phase2a.py <worktree_root>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).resolve()
    sb = root / "artifacts" / "research" / "strike_baseline"
    ledger = pd.read_parquet(sb / LEDGER_RUN / "ledger.parquet")
    ok = ledger[ledger["status"] == "ok"].copy()
    ok["y"] = (ok["outcome"] == "YES").astype(int)
    ok["close"] = pd.to_datetime(ok["end_time_est"], utc=True)

    # Sequential market splits (frozen 60/20/20).
    mk = ok[["market_id", "close"]].drop_duplicates().sort_values("close").reset_index(drop=True)
    n = len(mk)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    mk["split"] = np.where(mk.index < i1, "train", np.where(mk.index < i2, "validation", "test"))
    ok = ok.merge(mk[["market_id", "split"]], on="market_id", how="left")
    assert ok["split"].notna().all() and ok.groupby("market_id")["split"].nunique().max() == 1

    tr, va, te = (ok[ok["split"] == s] for s in ("train", "validation", "test"))
    train_rate = float(tr["y"].mean())

    # M1 fit on TRAIN only.
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000, random_state=0))
    pipe.fit(tr[FEATS].to_numpy(dtype=float), tr["y"].to_numpy())

    probs = {
        "M0": ok["p0"].to_numpy(dtype=float),
        "M1": pipe.predict_proba(ok[FEATS].to_numpy(dtype=float))[:, 1],
        "C": np.full(len(ok), train_rate),
    }

    metrics: dict = {
        "protocol_version": "1.2",
        "input_ledger": LEDGER_RUN,
        "n_ok": int(len(ok)),
        "splits": {s: {"markets": int((mk['split'] == s).sum()),
                       "rows": int((ok['split'] == s).sum()),
                       "close_min": str(mk[mk['split'] == s]['close'].min()),
                       "close_max": str(mk[mk['split'] == s]['close'].max())}
                   for s in ("train", "validation", "test")},
        "train_yes_rate": round(train_rate, 6),
        "models": {},
    }
    rel_frames = []
    for mname, p in probs.items():
        mrec: dict = {"prob": {}, "econ": {}}
        for s in ("train", "validation", "test"):
            m = (ok["split"] == s).to_numpy()
            y = ok.loc[m, "y"].to_numpy(dtype=float)
            pp = p[m]
            rel = reliability(y, pp)
            rel["model"] = mname
            rel["split"] = s
            rel_frames.append(rel)
            mrec["prob"][s] = {"n": int(m.sum()), "yes_rate": round(float(y.mean()), 6),
                               "brier": round(brier(y, pp), 6),
                               "log_loss": round(logloss(y, pp), 6),
                               "ece_10": round(ece_from_table(rel, int(m.sum())), 6)}
            mrec["econ"][s] = economic(ok.loc[m].reset_index(drop=True), pp)
        metrics["models"][mname] = mrec

    run_id = f"sb2a_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out = sb / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / "phase2a_metrics.json",
                  json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    rel_all = pd.concat(rel_frames, ignore_index=True)
    write_text_lf(out / "reliability.csv",
                  rel_all.to_csv(index=False, lineterminator="\n"))
    write_text_lf(out / "splits.csv",
                  mk.to_csv(index=False, lineterminator="\n"))

    proto = root / "research" / "strike_baseline" / "protocol.yaml"
    payload = {
        "run_id": run_id, "phase": "2a",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(root), "protocol_sha256": sha256_file(proto),
        "input_ledger": LEDGER_RUN,
        "ledger_sha256": sha256_file(sb / LEDGER_RUN / "ledger.parquet"),
        "artifacts": {},
    }
    for pth in sorted(out.iterdir()):
        if pth.is_file():
            payload["artifacts"][pth.name] = {"sha256": sha256_file(pth), "bytes": pth.stat().st_size}
    write_manifest(out / "run_manifest.json", payload)

    print(f"\nRUN_COMPLETE dir={out}")
    for mname in ("M0", "M1", "C"):
        t = metrics["models"][mname]["prob"]["test"]
        print(f"  {mname} test: brier={t['brier']} logloss={t['log_loss']} ece={t['ece_10']} n={t['n']}")


if __name__ == "__main__":
    main()
