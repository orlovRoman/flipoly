"""targets.py v1.0.0 — native/contract targets, horizons, folds, regimes."""
import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_s(s):
    """Resolution-agnostic epoch seconds (float)."""
    v = (s - _EPOCH) // pd.Timedelta("1s")
    try:
        return v.astype(np.float64)
    except AttributeError:
        return float(v)

SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
              "DOGE": "DOGEUSDT", "XRP": "XRPUSDT"}
FOLD_RANGES = [("F1", "2026-07-27", "2026-08-03"), ("F2", "2026-08-03", "2026-08-10"),
               ("F3", "2026-08-10", "2026-08-17"), ("F4", "2026-08-17", "2026-08-24"),
               ("F5", "2026-08-24", "2026-08-31"), ("F6", "2026-09-01", "2026-09-08")]
TRAINPOOL_LT = "2026-07-27"


def assign_fold(end):
    if pd.isna(end):
        return "unknown"
    e = str(end)[:10]
    for n, lo, hi in FOLD_RANGES:
        if lo <= e < hi:
            return n
    if e < TRAINPOOL_LT:
        return "trainpool"
    return "gap"


def build_candle_arrays(candles):
    out = {}
    for (sym, iv), g in candles.groupby(["symbol", "interval"], sort=True):
        g = g.sort_values("open_time", kind="mergesort").reset_index(drop=True)
        ot = np.asarray(epoch_s(g["open_time"]), dtype=np.float64)
        ct = np.asarray(epoch_s(g["close_time"]), dtype=np.float64)
        out[(sym, iv)] = {"ot": ot, "ct": ct,
                          "o": g["open"].to_numpy(dtype=np.float64),
                          "c": g["close"].to_numpy(dtype=np.float64)}
    return out


def legacy_native_for(df, carr):
    """Next-candle direction per row (NaN where unavailable). For decision at
    dec: first candle with open_time > dec on the asset's 15m grid; requires
    grid continuity (open - prev_close within [0, 1]s); return = close/open-1.
    Mirrors legacy training label ret_1.shift(-1) > 0 (future data legitimate
    for labels only, never for features)."""
    n = len(df)
    nat = np.full(n, np.nan)
    nend = np.full(n, np.nan)
    syms = df["asset"].map(SYMBOL_MAP)
    dec = np.asarray(epoch_s(df["decision_at"]), dtype=np.float64)
    for sym, idx in pd.Series(np.arange(n)).groupby(syms, sort=True):
        a = carr.get((sym, "15m"))
        if a is None:
            continue
        ii = idx.to_numpy()
        d = dec[ii]
        k = np.searchsorted(a["ot"], d, side="right")
        ok = (k < len(a["ot"])) & (k > 0)
        gap = np.full(ii.shape, np.inf)
        gap[ok] = a["ot"][k[ok]] - a["ct"][k[ok] - 1]
        ok = ok & (gap >= 0.0) & (gap <= 1.0)
        kk = k[ok]
        r = a["c"][kk] / a["o"][kk] - 1.0
        good = np.isfinite(r) & (a["o"][kk] > 0)
        sel = ii[ok][good]
        nat[sel] = (r[good] > 0).astype(np.float64)
        nend[sel] = a["ct"][kk[good]]
    return nat, nend
