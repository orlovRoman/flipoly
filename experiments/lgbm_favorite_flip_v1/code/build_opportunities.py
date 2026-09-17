"""build_opportunities.py — FLIP_OPPORTUNITIES.parquet (spec v1.0.0).

Unit: market x window (T-10/T-5/T-2). Causal only:
  window snapshot = latest series row with recorded_at <= decision_at
  anchor event    = latest decision event with decision_at <= decision_at,
                    age <= 120s, flip_native + p_market_yes present (v1.0.0)
  funnel match    = latest funnel row with created_at <= decision_at,
                    age <= 300s, p_flip present
Target favorite_flip from snapshot (yes_mid vs final_outcome), cross-checked
against anchor.flip_native and contract_target vs series outcome.
No model training here. Outputs to D:\\lgbm-favorite-flip-v1\\out\\.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import candidate_side, favorite_flip, favorite_side, market_flip

AUDIT = r"D:\lgbm-audit-v1"
OUTDIR = r"D:\lgbm-favorite-flip-v1\out"
WINDOWS = (("T-10", 600), ("T-5", 300), ("T-2", 120))
ANCHOR_MAX_AGE = 120
FUNNEL_MAX_AGE = 300

FOLD_DEF = [
    ("SEED", "2026-08-03", "2026-08-07"),
    ("F1", "2026-08-07", "2026-08-11"),
    ("F2", "2026-08-11", "2026-08-15"),
    ("F3", "2026-08-15", "2026-08-19"),
    ("F4", "2026-08-19", "2026-08-23"),
    ("F5", "2026-08-23", "2026-08-27"),
    ("F6", "2026-08-27", "2026-08-31"),
]
FINAL_FROM = "2026-08-31"


def fold_of(end_day):
    if end_day >= FINAL_FROM:
        return "FINAL_LOCKED"
    for name, lo, hi in FOLD_DEF:
        if lo <= end_day < hi:
            return name
    return "FOLD_GAP"


def load_events():
    d = pd.read_parquet(
        os.path.join(AUDIT, "out", "shared_ds", "shared_dataset.parquet"),
        columns=["decision_event_id", "decision_at", "market_id", "asset",
                 "market_end", "p_market_yes", "yes_ask", "no_ask",
                 "synthetic_no", "quote_ok", "contract_target", "flip_native"])
    d["market_id"] = d["market_id"].astype(str)
    d["decision_at"] = pd.to_datetime(d["decision_at"], utc=True)
    d["market_end"] = pd.to_datetime(d["market_end"], utc=True)
    return d


def load_series():
    parts = []
    for fp in sorted(glob.glob(os.path.join(AUDIT, "data", "exp", "snapsdec",
                                            "snapsdec_*.csv.gz"))):
        s = pd.read_csv(fp, usecols=["market_id", "recorded_at", "mid_price",
                                     "best_bid", "best_ask", "spread",
                                     "price_velocity", "volume_5min",
                                     "final_outcome"],
                        dtype={"market_id": str}, low_memory=False)
        s["recorded_at"] = pd.to_datetime(s["recorded_at"], utc=True,
                                          format="mixed")
        parts.append(s)
    fl = pd.read_csv(os.path.join(AUDIT, "data", "exp", "flipslice",
                                  "flipslice.csv.gz"),
                     usecols=["market_id", "recorded_at", "mid_price",
                              "best_bid", "best_ask", "spread",
                              "price_velocity", "volume_5min",
                              "final_outcome"],
                     dtype={"market_id": str}, low_memory=False,
                     on_bad_lines="skip")
    fl["recorded_at"] = pd.to_datetime(fl["recorded_at"], utc=True,
                                       format="mixed")
    parts.append(fl)
    s = pd.concat(parts, ignore_index=True)
    s = s.dropna(subset=["recorded_at"]).sort_values(
        ["market_id", "recorded_at"], kind="mergesort")
    s = s.drop_duplicates(subset=["market_id", "recorded_at"], keep="last")
    return s


def load_funnel():
    parts = []
    for fp in sorted(glob.glob(os.path.join(AUDIT, "data", "exp", "funnel",
                                            "funnel_*.csv.gz"))):
        f = pd.read_csv(fp, usecols=["market_id", "created_at", "p_flip",
                                     "entry_model_key", "entry_model_version",
                                     "entry_model_ece"],
                        dtype={"market_id": str}, low_memory=False)
        f["created_at"] = pd.to_datetime(f["created_at"], utc=True,
                                         format="mixed")
        f = f[f["p_flip"].notna()]
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    f = f.dropna(subset=["created_at"]).sort_values(
        ["market_id", "created_at"], kind="mergesort")
    return f


def load_entry_types():
    inv = pd.read_csv(os.path.join(AUDIT, "out", "MODEL_INVENTORY.csv"),
                      dtype={"id": str}, low_memory=False,
                      usecols=["asset", "version", "model_type"])
    m = {}
    for (k, v), g in inv.groupby(["asset", "version"]):
        types = set(g["model_type"].tolist())
        m[(str(k), str(v))] = ("logreg" if types == {"logreg"}
                               else ("mixed" if "logreg" in types
                                     else "nonlogreg"))
    return m


def ver_key(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return None if pd.isna(v) else str(v)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("loading events...", flush=True)
    ev = load_events()
    print("loading series...", flush=True)
    se = load_series()
    print("loading funnel...", flush=True)
    fu = load_funnel()
    entry_types = load_entry_types()

    ev_by_mk = {mk: g.sort_values("decision_at").reset_index(drop=True)
                for mk, g in ev.groupby("market_id", sort=True)}
    se_by_mk = {mk: g.sort_values("recorded_at").reset_index(drop=True)
                for mk, g in se.groupby("market_id", sort=True)}
    fu_by_mk = {mk: g.sort_values("created_at").reset_index(drop=True)
                for mk, g in fu.groupby("market_id", sort=True)}

    uni = ev.groupby("market_id").agg(
        asset=("asset", "first"),
        end=("market_end", lambda s: s.mode().iloc[0] if s.notna().any()
             else pd.NaT),
        ct_set=("contract_target", lambda s: tuple(sorted(
            v for v in s.dropna().unique().tolist()))))
    uni["end_day"] = uni["end"].dt.strftime("%Y-%m-%d")
    uni["fold"] = uni["end_day"].map(
        lambda e: fold_of(e) if isinstance(e, str) else "FOLD_GAP")

    market_block = {}  # mk -> block reason (mismatch)
    rows = []
    stats = {"markets_total": len(uni), "markets_resolved": 0,
             "markets_unresolved": 0, "markets_conflict": 0,
             "markets_no_end": 0, "markets_final_locked": 0,
             "markets_fold_gap": 0, "markets_dev": 0}

    for mk, u in uni.iterrows():
        if pd.isna(u["end"]):
            stats["markets_no_end"] += 1
            continue
        if len(u["ct_set"]) == 0:
            stats["markets_unresolved"] += 1
            continue
        if len(u["ct_set"]) > 1:
            stats["markets_conflict"] += 1
            continue
        stats["markets_resolved"] += 1
        fold = u["fold"]
        if fold == "FINAL_LOCKED":
            stats["markets_final_locked"] += 1
            continue
        if fold == "FOLD_GAP":
            stats["markets_fold_gap"] += 1
            continue
        stats["markets_dev"] += 1
        ct = float(u["ct_set"][0])
        ct_out = "YES" if ct == 1.0 else "NO"
        end = u["end"]
        eseg = se_by_mk.get(mk)
        eeg = ev_by_mk.get(mk)
        fug = fu_by_mk.get(mk)
        # series outcome consistency (single outcome per market)
        if eseg is not None:
            so = set(v for v in eseg["final_outcome"].dropna().unique()
                     if v in ("YES", "NO"))
            if len(so) > 1:
                market_block[mk] = "OUTCOME_MISMATCH"
            elif len(so) == 1 and so.pop() != ct_out:
                market_block[mk] = "OUTCOME_MISMATCH"
        for wname, wsec in WINDOWS:
            dt = end - pd.Timedelta(seconds=wsec)
            rec = {"opportunity_id": "%s@%s" % (mk, wname),
                   "market_id": mk, "asset": u["asset"],
                   "decision_window": wname, "decision_at": dt,
                   "market_end_at": end, "fold": fold,
                   "reason_code": "OK"}
            # --- window snapshot (causal) ---
            snap = None
            if eseg is not None:
                pos = eseg["recorded_at"].searchsorted(dt, side="right") - 1
                if pos >= 0:
                    snap = eseg.iloc[pos]
            if snap is None:
                rec["reason_code"] = "NO_CAUSAL_SNAPSHOT"
                rows.append(rec)
                continue
            out = snap["final_outcome"]
            ym = snap["mid_price"]
            rec["snapshot_age_seconds"] = float(
                (dt - snap["recorded_at"]).total_seconds())
            rec["snap_recorded_at"] = snap["recorded_at"]
            rec["snap_best_bid"] = snap["best_bid"]
            rec["snap_best_ask"] = snap["best_ask"]
            rec["snap_spread"] = snap["spread"]
            rec["snap_price_velocity"] = snap["price_velocity"]
            rec["snap_volume_5min"] = snap["volume_5min"]
            if out not in ("YES", "NO"):
                rec["reason_code"] = "BAD_OUTCOME"
                rows.append(rec)
                continue
            try:
                y = float(ym)
            except (TypeError, ValueError):
                rec["reason_code"] = "BAD_YES_MID"
                rows.append(rec)
                continue
            if not np.isfinite(y) or y <= 0.0 or y >= 1.0:
                rec["reason_code"] = "BAD_YES_MID"
                rows.append(rec)
                continue
            if y == 0.5:
                rec["reason_code"] = "AMBIGUOUS_FAVORITE"
                rows.append(rec)
                continue
            fav = "YES" if y > 0.5 else "NO"
            cand = candidate_side(fav)
            rec.update(yes_mid=y, favorite_side=fav,
                       candidate_side=cand, final_outcome=out,
                       favorite_flip=favorite_flip(fav, out))
            om, oprov = market_flip(y, fav)
            rec.update(outsider_mid=om, outsider_mid_provenance=oprov,
                       p_market_flip=om)
            # --- anchor event (causal, fresh, quoted) ---
            anch = None
            if eeg is not None:
                c = eeg[(eeg["decision_at"] <= dt)
                        & eeg["flip_native"].notna()
                        & eeg["p_market_yes"].notna()]
                if len(c):
                    anch = c.iloc[-1]
            if anch is None:
                rec["reason_code"] = "NO_ANCHOR_EVENT"
                rows.append(rec)
                continue
            age = float((dt - anch["decision_at"]).total_seconds())
            rec["anchor_age_seconds"] = age
            rec["anchor_event_id"] = int(anch["decision_event_id"])
            if age > ANCHOR_MAX_AGE:
                rec["reason_code"] = "NO_FRESH_ANCHOR"
                rows.append(rec)
                continue
            # v1.1.0: flip cross-check is diagnostic (leadership flips in the
            # anchor->snapshot gap are the studied phenomenon, not data error)
            rec["flipnative_agree"] = bool(
                int(anch["flip_native"]) == rec["favorite_flip"])
            rec["side_agree"] = bool(
                (float(anch["p_market_yes"]) > 0.5) == (fav == "YES"))
            ask = (anch["yes_ask"] if cand == "YES" else anch["no_ask"])
            if pd.isna(ask) or not (0.0 < float(ask) < 1.0):
                rec["econ_available"] = False
                rec["candidate_ask"] = np.nan
                rec["candidate_ask_provenance"] = "ASK_MISSING"
            else:
                rec["econ_available"] = True
                rec["candidate_ask"] = float(ask)
                synth = bool(anch["synthetic_no"]) and cand == "NO"
                rec["candidate_ask_provenance"] = ("ASK_SYNTHETIC" if synth
                                                  else "ASK_OBSERVED")
            rec["anchor_p_market_yes"] = float(anch["p_market_yes"])
            # --- funnel LogReg (causal); flag if unavailable ---
            rec["p_logreg_flip"] = np.nan
            rec["logreg_available"] = False
            if fug is not None:
                c = fug[fug["created_at"] <= dt]
                if len(c):
                    fr = c.iloc[-1]
                    fage = float((dt - fr["created_at"]).total_seconds())
                    key = (fr["entry_model_key"]
                           if isinstance(fr["entry_model_key"], str) else None)
                    et = entry_types.get((key, ver_key(
                        fr["entry_model_version"])))
                    if fage <= FUNNEL_MAX_AGE and et == "logreg":
                        rec["p_logreg_flip"] = float(fr["p_flip"])
                        rec["logreg_model_key"] = key
                        rec["logreg_model_version"] = fr["entry_model_version"]
                        rec["logreg_entry_ece"] = fr["entry_model_ece"]
                        rec["logreg_age_seconds"] = fage
                        rec["logreg_available"] = True
            rows.append(rec)

    op = pd.DataFrame(rows)
    # market-level OUTCOME_MISMATCH block only (genuinely market-level:
    # all rows' labels suspect). v1.1.0: no flipnative market block.
    if market_block:
        blk = op["market_id"].map(market_block)
        op.loc[blk.notna(), "reason_code"] = ("MARKET_BLOCKED_"
                                              + blk[blk.notna()])
    # v1.1.0 subset flags (rows stay in OK universe)
    for col in ("flipnative_agree", "econ_available", "logreg_available"):
        if col not in op.columns:
            op[col] = False if col != "flipnative_agree" else np.nan
    op["out_of_band"] = False
    if "outsider_mid" in op.columns:
        ob = (op["reason_code"] == "OK") & (
            op["outsider_mid"].isna() | (op["outsider_mid"] < 0.05)
            | (op["outsider_mid"] > 0.50))
        op.loc[ob, "out_of_band"] = True

    op.to_parquet(os.path.join(OUTDIR, "FLIP_OPPORTUNITIES.parquet"),
                  index=False)
    lad = op["reason_code"].value_counts().to_dict()
    lad["_markets"] = stats
    with open(os.path.join(OUTDIR, "COVERAGE_LADDER.json"), "w") as f:
        json.dump(lad, f, indent=1, default=str)
    ok = op[op["reason_code"] == "OK"]
    print("rows=%d OK=%d (%.3f)" % (len(op), len(ok),
                                    len(ok) / max(len(op), 1)))
    print("reason counts:")
    for k, v in sorted(lad.items(), key=lambda kv: -kv[1]
                       if isinstance(kv[1], int) else 0):
        if not k.startswith("_"):
            print("  %s: %d" % (k, v))
    print("markets:", json.dumps(stats, indent=1))
    if len(ok):
        print("OK by fold: %s" % ok["fold"].value_counts().to_dict())
        print("OK by window: %s"
              % ok["decision_window"].value_counts().to_dict())
        print("OK by asset: %s" % ok["asset"].value_counts().to_dict())
        print("OK flip rate: %.4f" % ok["favorite_flip"].mean())
        print("OK fav YES frac: %.4f"
              % (ok["favorite_side"] == "YES").mean())
        print("OK flipnative_agree rate: %.4f"
              % ok["flipnative_agree"].fillna(False).mean())
        print("OK out_of_band: %d" % int(ok["out_of_band"].sum()))
        print("OK logreg paired: %d (%.3f)"
              % (int(ok["logreg_available"].sum()),
                 ok["logreg_available"].mean()))
        print("OK econ (ask): %d (%.3f)"
              % (int(ok["econ_available"].sum()),
                 ok["econ_available"].mean()))
        print("OK flip rate by window:")
        print(ok.groupby("decision_window")["favorite_flip"].agg(
            ["mean", "count"]).to_string())
        print("OK flip rate by fold:")
        print(ok.groupby("fold")["favorite_flip"].agg(
            ["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
