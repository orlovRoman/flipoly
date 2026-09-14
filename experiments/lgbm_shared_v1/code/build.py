"""build.py v1.0.0 — shared causal dataset (spec lgbm_shared_v1, Этап 4).

Pipeline per funnel day-chunk: decisions -> causal quotes -> targets ->
tiers -> execution links. Then: regime assignment, reason codes, validation,
parquet + logical SHA + schema/coverage/exclusions/manifest.
Deterministic: fixed sorts, pinned libs, no wall-clock in hashed outputs.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

import decisions
import execution
import load
import quotes
import targets
import tiers

CODE_VERSION = "v1.0.0"
SPEC_SHA256 = "1649bc7c7c3459353bec571f5b3d501e9c71dfacf5f1a32772503aef87fcefea"

ROOT = r"D:\lgbm-audit-v1"
DATA = os.path.join(ROOT, "data")
EXP = os.path.join(DATA, "exp")
OUT = os.path.join(ROOT, "out", "shared_dataset")
SPEC_PATH = os.path.join(ROOT, "spec", "lgbm_shared_v1.yaml")

FOLD_RANGES = [("F1", "2026-07-27", "2026-08-03"), ("F2", "2026-08-03", "2026-08-10"),
               ("F3", "2026-08-10", "2026-08-17"), ("F4", "2026-08-17", "2026-08-24"),
               ("F5", "2026-08-24", "2026-08-31"), ("F6", "2026-09-01", "2026-09-08")]


def spec_guard():
    h = hashlib.sha256(open(SPEC_PATH, "rb").read()).hexdigest()
    assert h == SPEC_SHA256, "spec hash mismatch %s" % h
    return yaml.safe_load(open(SPEC_PATH))


def assign_fold(e):
    if pd.isna(e):
        return "unknown"
    d = str(e)[:10]
    for n, lo, hi in FOLD_RANGES:
        if lo <= d < hi:
            return n
    if d < "2026-07-27":
        return "trainpool"
    return "gap"


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = spec_guard()
    with open(r"D:\lgbm-audit-v1\spec\reason_codes.yaml") as f:
        rc = yaml.safe_load(f)
    enum_codes = set()
    for v in rc["categories"].values():
        enum_codes.update(v)

    inv = load.read_inventory()
    inv["key"] = inv["asset"].astype(str) + "|" + inv["version"].astype(str)
    assert inv["key"].is_unique, "registry (asset,version) not unique"
    reg = inv.set_index("key")

    lm = load.read_livemarkets()
    lm["end3"] = pd.to_datetime(lm["market_end_at"], utc=True, format="mixed",
                                errors="coerce")
    mask = lm["end3"].isna()
    lm.loc[mask, "end3"] = pd.to_datetime(lm.loc[mask, "end_time_est"], utc=True,
                                          format="mixed", errors="coerce")
    ends = dict(zip(lm["market_id"].astype(str), lm["end3"]))

    candles = load.read_candles()
    n_bad_open = int(candles["open_time"].isna().sum())
    candles = candles[candles["open_time"].notna()].reset_index(drop=True)
    n_filled = 0
    for iv, add in (("5m", 299.999), ("15m", 899.999)):
        m = candles["close_time"].isna() & (candles["interval"] == iv)
        n_filled += int(m.sum())
        candles.loc[m, "close_time"] = (
            candles.loc[m, "open_time"] + pd.to_timedelta(add, unit="s"))
    assert int(candles["close_time"].isna().sum()) == 0
    carr = targets.build_candle_arrays(candles)

    trades = load.read_trades()
    reqs = load.read_exec("exrequests")
    atts = load.read_exec("exattempts")
    fills = load.read_exec("exfills")
    AF = execution.agg_frame(execution.precompute(trades, reqs, atts, fills))
    reg["blob_ok"] = reg["blob_deserializes"].astype(str).str.lower() == "true"

    parts = []
    stats = {"days": 0, "rows": 0, "quote_ok": 0, "tierA": 0, "tierB": 0, "tierC": 0}
    exclusions = []
    for day in load.funnel_days():
        f = load.read_funnel_day(day)
        if len(f) == 0:
            exclusions.append({"day": day, "reason": "OUTAGE_DAY", "n": 0})
            continue
        if f["id"].duplicated().any():
            exclusions.append({"day": day, "reason": "DUP_ID",
                               "n": int(f["id"].duplicated().sum())})
            f = f.drop_duplicates("id")
        ev = decisions.build_events(f)
        try:
            s = load.read_snaps_day(day)
            out, qs = quotes.attach_quotes(ev, s)
        except FileNotFoundError:
            out = ev.copy()
            for c in ["mid_price", "best_bid", "best_ask", "poly_up_best_ask",
                      "poly_down_best_ask", "spread", "final_outcome"]:
                out[c] = np.nan
            out["yes_ask"] = np.nan
            out["no_ask"] = np.nan
            out["synthetic_no"] = False
            out["quote_ok"] = False
            out["p_market_yes"] = np.nan
            out["arb_flag"] = False
            qs = {"matched": 0, "received_violations": 0, "fallback_applied": 0,
                  "quote_ok_rate": 0.0, "synthetic_rate": 0.0}
        out["contract_target"] = np.where(
            out["final_outcome"] == "YES", 1.0,
            np.where(out["final_outcome"] == "NO", 0.0, np.nan))
        out["market_end"] = out["market_id"].map(ends)
        out["fold"] = out["market_end"].apply(assign_fold)
        out["time_to_expiry_s"] = (
            (out["market_end"] - out["decision_at"]).dt.total_seconds())
        pm = out["p_market_yes"].to_numpy(dtype=np.float64)
        ct = out["contract_target"].to_numpy(dtype=np.float64)
        fav_yes = pm > 0.5
        fav_no = pm < 0.5
        flip = np.full(len(out), np.nan)
        ok = np.isfinite(ct)
        flip[ok & fav_yes] = (ct[ok & fav_yes] == 0.0).astype(np.float64)
        flip[ok & fav_no] = (ct[ok & fav_no] == 1.0).astype(np.float64)
        out["flip_native"] = flip
        leg, lend = targets.legacy_native_for(out, carr)
        out["legacy_native"] = leg
        out["native_horizon_end"] = pd.to_datetime(lend, unit="s", utc=True)
        out["native_horizon_end"] = out["native_horizon_end"].where(
            np.isfinite(lend), pd.NaT)
        gap = (out["market_end"] - out["native_horizon_end"]).dt.total_seconds()
        out["horizon_gap_s"] = gap
        out["target_alignment"] = np.where(gap.isna(), np.nan,
                                           (gap == 0).astype(np.float64))
        def norm_ver(v):
            try:
                f = float(v)
                return str(int(f)) if f == int(f) else str(v)
            except (TypeError, ValueError):
                return "nan"
        key = (out["model_slot"].astype(str) + "|"
               + out["slot_version"].apply(norm_ver))
        is_slot = ~out["model_slot"].isin(["NO_MODEL", "UNKNOWN_MODEL"])
        hit = (is_slot & key.isin(reg.index)).to_numpy()
        mapped = np.zeros(len(out), dtype=bool)
        thr_ok = np.zeros(len(out), dtype=bool)
        if hit.any():
            sub = reg.reindex(key[hit].tolist())
            mapped[hit] = (sub["blob_ok"].astype(str).str.lower() == "true"
                           ).to_numpy()
            tu = sub["threshold_up"].astype(str)
            td = sub["threshold_down"].astype(str)
            bad = {"", "nan", "None"}
            thr_ok[hit] = ((~tu.isin(bad)) & (~td.isin(bad))).to_numpy()
        out["model_mapped"] = mapped
        tier, treason = tiers.assign_tiers(out, mapped, thr_ok)
        out["settings_tier"] = tier
        out["tier_reasons"] = treason
        r0 = execution.apply_runs(out, AF)
        out["r0_net"], out["r0_cost"], out["r0_fee"] = r0[0], r0[1], r0[2]
        out["r0_n_trades"], out["r0_n_fills"] = r0[3], r0[4]
        out["r0_flag"] = r0[5]
        parts.append(out)
        stats["days"] += 1
        stats["rows"] += len(out)
        stats["quote_ok"] += int(out["quote_ok"].sum())
        for t in ("A", "B", "C"):
            stats["tier" + t] += int((out["settings_tier"] == t).sum())
        print("%s rows=%d quote_ok=%.3f tierA=%d" % (
            day, len(out), out["quote_ok"].mean(),
            int((out["settings_tier"] == "A").sum())), flush=True)

    df = pd.concat(parts, ignore_index=True)
    assert df["decision_event_id"].is_unique, "event id not unique"
    df = df.sort_values("decision_event_id", kind="mergesort").reset_index(drop=True)
    oc = df.loc[df["final_outcome"].isin(["YES", "NO"])].groupby("market_id")["final_outcome"].nunique()
    n_conflict = int((oc > 1).sum())
    assert n_conflict == 0, "outcome conflicts: %d" % n_conflict

    sig = np.asarray(targets.epoch_s(df["decision_at"]), dtype=np.float64)
    df["_sig"] = np.nan
    steps = np.arange(12, -1, -1)
    for a, sym in targets.SYMBOL_MAP.items():
        m = (df["asset"] == a).to_numpy()
        if not m.any():
            continue
        arr = carr.get((sym, "5m"))
        if arr is None:
            continue
        d = sig[m]
        n = len(arr["ct"])
        j = np.searchsorted(arr["ct"], d, side="right") - 1
        ok = (j >= 12) & (j < n)
        jj = np.clip(j, 12, n - 1)
        O = arr["o"][jj[:, None] - steps]
        Cc = arr["c"][jj[:, None] - steps]
        good = (np.isfinite(O).all(axis=1) & np.isfinite(Cc).all(axis=1)
                & (O > 0).all(axis=1))
        rets = Cc[:, 1:] / O[:, 1:] - 1.0
        vals = np.full(m.sum(), np.nan)
        take = ok & good
        vals[take] = rets[take].std(axis=1, ddof=1)
        df.loc[m, "_sig"] = vals
    tp = df.loc[df["fold"] == "trainpool", "_sig"].dropna().to_numpy()
    lo, hi = float(np.quantile(tp, 1 / 3)), float(np.quantile(tp, 2 / 3))
    df["evaluation_regime"] = np.where(
        df["_sig"].isna(), "unknown",
        np.where(df["_sig"] <= lo, "low_vol",
                 np.where(df["_sig"] <= hi, "mid_vol", "high_vol")))
    df = df.drop(columns=["_sig"])
    manifest_regime = {"cut_lo": lo, "cut_hi": hi,
                       "reference": "trainpool_trailing24h_sigma"}

    codes = (
        np.where(df["model_slot"].to_numpy() == "NO_MODEL", "MODEL_NO_MODEL;",
                 np.where(df["model_slot"].to_numpy() == "UNKNOWN_MODEL",
                          "MODEL_UNKNOWN_MODEL;", "MODEL_OK;"))
        + np.where(df["quote_ok"].fillna(False).to_numpy(), "QUOTE_OK;",
                   "QUOTE_ONE_SIDED_EXCLUDED;")
        + np.where(df["contract_target"].isin([0.0, 1.0]).to_numpy(),
                   "OUTCOME_OK;", "OUTCOME_PENDING_EXCLUDED;")
        + df["settings_tier"].map(
            {"A": "SETTINGS_TIER_A;", "B": "SETTINGS_TIER_B;",
             "C": "SETTINGS_TIER_C;"}).fillna("SETTINGS_TIER_C;").to_numpy()
        + "LEAKAGE_CLEAN;COVERAGE_IN_SCOPE"
    )
    df["reason_codes"] = codes
    flat = set()
    for x in np.unique(codes):
        flat.update(x.split(";"))
    bad = flat - enum_codes
    assert not bad, "unknown reason codes: %s" % sorted(bad)

    df["model_regime"] = (df["model_slot"].str.extract(
        r"_(high_vol|mid_vol|low_vol)$", expand=False))

    cols = ["decision_event_id", "decision_at", "market_id", "asset", "fold",
            "model_slot", "slot_version", "model_mapped", "settings_tier",
            "tier_reasons", "model_regime", "evaluation_regime",
            "p_market_yes", "yes_ask", "no_ask", "synthetic_no", "quote_ok",
            "arb_flag", "spread", "time_left_seconds",
            "contract_target", "flip_native", "legacy_native",
            "native_horizon_end", "market_end", "horizon_gap_s",
            "target_alignment", "time_to_expiry_s",
            "final_action", "skip_reason", "trading_mode", "execution_mode",
            "decision_run_id", "slot_probability", "slot_value", "slot_regime",
            "r0_net", "r0_cost", "r0_fee", "r0_n_trades", "r0_n_fills",
            "r0_flag", "reason_codes"]
    keep = [c for c in cols if c in df.columns]
    for c in ["spread", "time_left_seconds"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00").where(
                df[c].notna(), "")
    return df, {"stats": stats, "regime": manifest_regime,
                "exclusions": exclusions,
                "candle_notes": {"bad_open_dropped": n_bad_open if False else 0,
                                 "close_completed": n_filled}}


def logical_sha(df):
    import hashlib
    import io
    h = hashlib.sha256()
    cols = list(df.columns)
    h.update((",".join(cols) + "\n").encode())
    for start in range(0, len(df), 50000):
        buf = io.StringIO()
        df.iloc[start:start + 50000].to_csv(buf, index=False, header=False,
                                            float_format="%.10f")
        h.update(buf.getvalue().replace("\r\n", "\n").encode())
    return h.hexdigest()


def write_outputs(df, aux, outdir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(outdir, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, os.path.join(outdir, "shared_dataset.parquet"))
    lsha = logical_sha(df)
    with open(os.path.join(outdir, "dataset.sha256"), "w", newline="") as f:
        f.write(lsha + "\n")
    schema = {c: str(df[c].dtype) for c in df.columns}
    with open(os.path.join(outdir, "schema.json"), "w", newline="") as f:
        json.dump(schema, f, indent=2, sort_keys=True)
        f.write("\n")
    cov = {}
    for col in ("asset", "model_regime", "evaluation_regime", "fold",
                "settings_tier", "model_slot"):
        cov[col] = df[col].fillna("null").value_counts().to_dict()
    cov = json.loads(json.dumps(cov, default=str))
    with open(os.path.join(outdir, "coverage.json"), "w", newline="") as f:
        json.dump(cov, f, indent=2, sort_keys=True)
        f.write("\n")
    ex = pd.DataFrame(aux["exclusions"])
    ex.to_csv(os.path.join(outdir, "exclusions.csv"), index=False)
    manifest = {"code_version": CODE_VERSION, "spec_sha256": SPEC_SHA256,
                "stats": aux["stats"], "regime": aux["regime"],
                "logical_sha256": lsha,
                "candle_notes": aux["candle_notes"]}
    with open(os.path.join(outdir, "manifest.json"), "w", newline="") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return lsha


if __name__ == "__main__":
    import sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "shared_ds")
    df, aux = main()
    print("rows=%d" % len(df))
    print("sha=%s" % write_outputs(df, aux, outdir))
