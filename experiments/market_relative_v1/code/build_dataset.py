"""feature_builder v1.0.0 — market_relative_v1 dataset build (spec v1.0.2).

Deterministic: markets processed in sorted order, all selections by
(recorded_at, id) / (close_time) ordering, no randomness, no wall-clock
in hashed outputs. Floats written with %.10f.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

CODE_VERSION = "v1.0.2"
SPEC_VERSION = "1.0.4"
SPEC_SHA256 = "2dce310ce3d4917061109d154520b0eb932dc9894b88b787d00fd32b0898071c"
CLOSE_FILL = {"5m": 299.999, "15m": 899.999}
# Cross-platform determinism: all text outputs use LF only (newline="").

WIN_OPEN = 330.0
WIN_CLOSE = 270.0
LOOKBACKS = (90, 180, 300)
LOOK_TOL = 60.0
CANDLE_5M_MAX_AGE = 330.0
CANDLE_15M_MAX_AGE = 930.0
SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
              "DOGE": "DOGEUSDT", "XRP": "XRPUSDT"}
ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP"]
FEATURES = ["p_market_yes", "spread", "dprob_90s", "dprob_180s", "dprob_300s",
            "prob_vel_90s", "time_to_expiry_s", "volume_5min", "price_velocity",
            "ret_5m_1", "ret_15m_1", "vol_5m_12", "candle_vol_5m_1",
            "taker_buy_share_5m_1", "breadth_ret_5m", "btc_ret_5m",
            "btc_vol_5m_12", "trend_15m"]
FLAGS = [f + "_m" for f in FEATURES]
DEV_FROM_EP = np.datetime64("2026-07-06T00:00:00").astype("datetime64[s]").astype(np.float64)
DEV_TO_EP = np.datetime64("2026-09-12T00:00:00").astype("datetime64[s]").astype(np.float64)
FOLD_EPS = [("F1", "2026-07-27", "2026-08-03"), ("F2", "2026-08-03", "2026-08-10"),
            ("F3", "2026-08-10", "2026-08-17"), ("F4", "2026-08-17", "2026-08-24"),
            ("F5", "2026-08-24", "2026-08-31"), ("F6", "2026-09-01", "2026-09-08")]
FOLD_EPS = [(n, np.datetime64(l).astype("datetime64[s]").astype(np.float64),
             np.datetime64(h).astype("datetime64[s]").astype(np.float64)) for n, l, h in FOLD_EPS]
TRAINPOOL_LT = float(np.datetime64("2026-07-27").astype("datetime64[s]").astype(np.float64))
FOLDS = [("F1", "2026-07-27", "2026-08-03"), ("F2", "2026-08-03", "2026-08-10"),
         ("F3", "2026-08-10", "2026-08-17"), ("F4", "2026-08-17", "2026-08-24"),
         ("F5", "2026-08-24", "2026-08-31"), ("F6", "2026-09-01", "2026-09-08")]


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_of(s):
    """Resolution-agnostic epoch seconds as float (works for us/ns Series/scalars)."""
    v = (s - _EPOCH) // pd.Timedelta("1s")
    try:
        return v.astype(np.float64)
    except AttributeError:
        return float(v)


def valid_px(x):
    return x == x and 0.0 < x < 1.0


def pick_prior(row):
    """p_market_yes: poly_up_mid fallback mid_price. Returns (p, ok)."""
    for k in ("poly_up_mid", "mid_price"):
        v = row[k]
        if valid_px(v):
            return v, True
    return np.nan, False


def pick_asks(row):
    """Returns (yes_ask, no_ask, synthetic_no, ok)."""
    ya = row["poly_up_best_ask"]
    if not valid_px(ya):
        ya = row["best_ask"]
    dn = row["poly_down_best_ask"]
    synth = False
    if not valid_px(dn):
        synth = True
        bb = row["best_bid"]
        dn = 1.0 - bb if bb == bb else np.nan
    if not valid_px(ya) or not valid_px(dn):
        return np.nan, np.nan, synth, False
    return ya, dn, synth, True


def assign_fold(end_ep):
    for name, lo, hi in FOLD_EPS:
        if lo <= end_ep < hi:
            return name
    if end_ep < TRAINPOOL_LT:
        return "trainpool"
    return "gap"


def load_inputs(indir):
    markets = pd.read_csv(os.path.join(indir, "markets.csv"), dtype={"market_id": str})
    snaps = pd.read_csv(os.path.join(indir, "snaps.csv"),
                        dtype={"market_id": str, "received_timestamp": str},
                        low_memory=False)
    candles = pd.read_csv(os.path.join(indir, "candles.csv"))
    for df, col in ((markets, "end3"), (snaps, "recorded_at")):
        df[col] = pd.to_datetime(df[col], utc=True)
    snaps["recvd"] = pd.to_datetime(snaps["received_timestamp"], utc=True)
    for col in ("open_time", "close_time"):
        candles[col] = pd.to_datetime(candles[col], utc=True, errors="coerce")
    n_bad_open = int(candles["open_time"].isna().sum())
    candles = candles[candles["open_time"].notna()].reset_index(drop=True)
    candles["_close_completed"] = False
    n_filled = 0
    for iv, add in CLOSE_FILL.items():
        mask = candles["close_time"].isna() & (candles["interval"] == iv)
        n_filled += int(mask.sum())
        candles.loc[mask, "close_time"] = (
            candles.loc[mask, "open_time"] + pd.to_timedelta(add, unit="s"))
        candles.loc[mask, "_close_completed"] = True
    candle_notes = {"bad_open_dropped": n_bad_open, "close_completed": n_filled}
    return markets, snaps, candles, candle_notes


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    markets, snaps, candles, candle_notes = load_inputs(indir)
    input_hashes = {n: sha_of(os.path.join(indir, n))
                    for n in ("markets.csv", "snaps.csv", "candles.csv")}

    snap_groups = {}
    for mid, g in snaps.groupby("market_id", sort=True):
        g = g.sort_values(["recorded_at", "id"], kind="mergesort").reset_index(drop=True)
        snap_groups[mid] = g

    cand = {}
    for (sym, iv), g in candles.groupby(["symbol", "interval"], sort=True):
        g = g.sort_values("close_time", kind="mergesort").reset_index(drop=True)
        cand[(sym, iv)] = {
            "ct": np.asarray(epoch_of(g["close_time"]), dtype=np.float64),
            "o": g["open"].to_numpy(dtype=np.float64),
            "c": g["close"].to_numpy(dtype=np.float64),
            "v": g["volume"].to_numpy(dtype=np.float64),
            "tb": g["taker_buy_volume"].to_numpy(dtype=np.float64),
        }

    def candle_ret(sym, iv, dec_ep, max_age):
        a = cand.get((sym, iv))
        if a is None:
            return np.nan, False
        i = int(np.searchsorted(a["ct"], dec_ep, side="right")) - 1
        if i < 0 or dec_ep - a["ct"][i] > max_age:
            return np.nan, False
        o, c = a["o"][i], a["c"][i]
        if not (o == o and c == c and o > 0):
            return np.nan, False
        return c / o - 1.0, True

    def candle_vol(sym, dec_ep):
        a = cand.get((sym, "5m"))
        if a is None:
            return np.nan, False
        i = int(np.searchsorted(a["ct"], dec_ep, side="right")) - 1
        if i < 12 or dec_ep - a["ct"][i] > CANDLE_5M_MAX_AGE:
            return np.nan, False
        o = a["o"][i - 12:i + 1]
        c = a["c"][i - 12:i + 1]
        if not np.all(np.isfinite(o)) or not np.all(np.isfinite(c)) or np.any(o <= 0):
            return np.nan, False
        rets = c / o - 1.0
        return float(np.std(rets, ddof=1)), True

    def candle_vol_share(sym, dec_ep):
        a = cand.get((sym, "5m"))
        if a is None:
            return (np.nan, False), (np.nan, False)
        i = int(np.searchsorted(a["ct"], dec_ep, side="right")) - 1
        if i < 0 or dec_ep - a["ct"][i] > CANDLE_5M_MAX_AGE:
            return (np.nan, False), (np.nan, False)
        v, tb = a["v"][i], a["tb"][i]
        vol_ok = bool(v == v and v > 0)
        if not vol_ok:
            return (np.nan, False), (np.nan, False)
        sh = tb / v if tb == tb else np.nan
        return (v, True), (sh, bool(sh == sh))

    out_rows = []
    excl = {}
    miss_count = {f: 0 for f in FEATURES}

    for mid in sorted(markets["market_id"].unique()):
        m = markets.loc[markets["market_id"] == mid].iloc[0]
        end3 = m["end3"]
        end_ep = float(epoch_of(end3))
        if not (DEV_FROM_EP <= end_ep < DEV_TO_EP):
            continue
        g = snap_groups.get(mid)
        if g is None:
            excl["NO_SNAPS"] = excl.get("NO_SNAPS", 0) + 1
            continue
        rec = np.asarray(epoch_of(g["recorded_at"]), dtype=np.float64)
        in_w = (rec >= end_ep - WIN_OPEN) & (rec <= end_ep - WIN_CLOSE)
        if not bool(np.any(in_w)):
            excl["MISSING_DECISION"] = excl.get("MISSING_DECISION", 0) + 1
            continue
        di = int(np.argmax(in_w))
        dec_ep = float(rec[di])
        drow = g.iloc[di]
        recvd = drow["recvd"]
        try:
            recvd_ep = float(epoch_of(recvd))
        except (TypeError, ValueError):
            recvd_ep = float("inf")
        if recvd_ep > dec_ep:
            excl["RECEIVED_GUARD"] = excl.get("RECEIVED_GUARD", 0) + 1
            continue
        tte = end_ep - dec_ep
        p, ok_p = pick_prior(drow)
        if not ok_p:
            excl["BAD_P"] = excl.get("BAD_P", 0) + 1
            continue
        sp = drow["spread"]
        if not (sp == sp):
            excl["BAD_SPREAD"] = excl.get("BAD_SPREAD", 0) + 1
            continue
        ya, na, synth, ok_a = pick_asks(drow)
        if not ok_a:
            excl["BAD_ASKS"] = excl.get("BAD_ASKS", 0) + 1
            continue
        if drow["best_bid"] > drow["best_ask"]:
            excl["CROSSED_BOOK"] = excl.get("CROSSED_BOOK", 0) + 1
            continue
        vol5 = drow["volume_5min"]
        pvel = drow["price_velocity"]

        def pat(N):
            lo = dec_ep - N - LOOK_TOL
            hi = dec_ep - N
            m2 = (rec <= hi) & (rec >= lo)
            if not bool(np.any(m2)):
                return np.nan, False
            j = int(np.max(np.nonzero(m2)[0]))
            r2 = g.iloc[j]
            pv, ok = pick_prior(r2)
            return (p - pv, True) if ok else (np.nan, False)

        d90, ok90 = pat(90)
        d180, ok180 = pat(180)
        d300, ok300 = pat(300)
        pvel90 = d90 / 90.0 if ok90 else np.nan

        per_asset = {}
        for a in ASSETS:
            sym = SYMBOL_MAP[a]
            r5, o5 = candle_ret(sym, "5m", dec_ep, CANDLE_5M_MAX_AGE)
            r15, o15 = candle_ret(sym, "15m", dec_ep, CANDLE_15M_MAX_AGE)
            vv, ov = candle_vol(sym, dec_ep)
            (cv, ocv), (tsh, otsh) = candle_vol_share(sym, dec_ep)
            per_asset[a] = (r5, o5, r15, o15, vv, ov, cv, ocv, tsh, otsh)
        my = per_asset[m["asset"]] if m["asset"] in per_asset else (np.nan,) * 10
        r5s = [per_asset[a][0] for a in ASSETS]
        r15s = [per_asset[a][2] for a in ASSETS]
        ok5 = [per_asset[a][1] for a in ASSETS]
        ok15 = [per_asset[a][3] for a in ASSETS]
        breadth = float(np.mean([v for v, o in zip(r5s, ok5) if o])) if sum(ok5) >= 3 else np.nan
        trend = float(np.mean([v for v, o in zip(r15s, ok15) if o])) if sum(ok15) >= 3 else np.nan
        btc = per_asset["BTC"]

        oc = m["outcome"]
        if oc == "YES":
            label = 1.0
        elif oc == "NO":
            label = 0.0
        else:
            excl["NO_LABEL"] = excl.get("NO_LABEL", 0) + 1
            continue
        fold = assign_fold(end_ep)
        if fold == "gap":
            excl["GAP_FOLD"] = excl.get("GAP_FOLD", 0) + 1
            continue

        feats = {
            "p_market_yes": (p, True), "spread": (sp, True),
            "dprob_90s": (d90, ok90), "dprob_180s": (d180, ok180),
            "dprob_300s": (d300, ok300), "prob_vel_90s": (pvel90, ok90),
            "time_to_expiry_s": (tte, True),
            "volume_5min": (vol5, bool(vol5 == vol5)),
            "price_velocity": (pvel, bool(pvel == pvel)),
            "ret_5m_1": (my[0], my[1]), "ret_15m_1": (my[2], my[3]),
            "vol_5m_12": (my[4], my[5]),
            "candle_vol_5m_1": (my[6], my[7]),
            "taker_buy_share_5m_1": (my[8], my[9]),
            "breadth_ret_5m": (breadth, bool(breadth == breadth)),
            "btc_ret_5m": (btc[0], btc[1]), "btc_vol_5m_12": (btc[4], btc[5]),
            "trend_15m": (trend, bool(trend == trend)),
        }
        row = {"market_id": mid, "asset": m["asset"], "end3": str(end3),
               "dec_at": str(drow["recorded_at"]), "tte_s": tte,
               "fold": fold, "label": label, "yes_ask": ya, "no_ask": na,
               "synthetic_no": bool(synth),
               "arb": bool(ya + na < 1.0), "outcome": oc}
        for f in FEATURES:
            v, ok = feats[f]
            row[f] = v if ok else np.nan
            row[f + "_m"] = 0.0 if ok else 1.0
            if not ok:
                miss_count[f] += 1
        out_rows.append(row)

    cols = (["market_id", "asset", "end3", "dec_at", "tte_s", "fold", "label"] +
            FEATURES + FLAGS + ["yes_ask", "no_ask", "synthetic_no", "arb", "outcome"])
    df = pd.DataFrame(out_rows, columns=cols).sort_values("market_id", kind="mergesort").reset_index(drop=True)
    ds_path = os.path.join(outdir, "dataset.csv")
    with open(ds_path, "w", newline="") as f:
        df.to_csv(f, index=False, float_format="%.10f", lineterminator="\n")
    ds_sha = sha_of(ds_path)

    folds = df["fold"].value_counts().to_dict()
    fold_cov = {}
    for name, _, _ in FOLDS:
        fold_cov[name] = int(folds.get(name, 0))
    manifest = {
        "code_version": CODE_VERSION, "spec_version": SPEC_VERSION,
        "spec_sha256": SPEC_SHA256, "input_hashes": input_hashes,
        "candle_completion": candle_notes,
        "dataset_sha256": ds_sha, "n_rows": int(len(df)),
        "exclusions": dict(sorted(excl.items())),
        "missing_rates": {f: miss_count[f] / len(df) if len(df) else 0.0 for f in FEATURES},
        "folds": dict(sorted(folds.items())),
        "label_balance": {str(k): int(v) for k, v in sorted(df["label"].value_counts().to_dict().items())},
        "synthetic_no_rate": float(df["synthetic_no"].mean()) if len(df) else 0.0,
        "arb_rate": float(df["arb"].mean()) if len(df) else 0.0,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", newline="") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(outdir, "dataset.sha256"), "w", newline="") as f:
        f.write(ds_sha + "\n")
    return manifest


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("MKTREL_ROOT", os.path.dirname(base))
    indir = os.environ.get("MKTREL_DATA", os.path.join(root, "data"))
    outdir = os.environ.get("MKTREL_BUILD", os.path.join(root, "build"))
    m = build(indir, outdir)
    print("rows=%d sha=%s" % (m["n_rows"], m["dataset_sha256"]))
    print("exclusions=%s" % (m["exclusions"],))
    print("folds=%s" % (m["folds"],))


if __name__ == "__main__":
    main()
