"""f0_feature_coverage.py — vectorized per-market as-of coverage."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAPS = os.path.join(ROOT, "data", "exp", "snapsdec")

ds = pd.read_parquet(
    os.path.join(ROOT, "out", "shared_ds", "shared_dataset.parquet"),
    columns=["decision_event_id", "decision_at", "market_id", "asset", "fold",
             "contract_target", "flip_native"],
)
ds["market_id"] = ds["market_id"].astype(str)
ds["decision_at"] = pd.to_datetime(ds["decision_at"], utc=True, errors="coerce")
ds = ds.dropna(subset=["decision_at", "market_id"]).reset_index(drop=True)
print("decisions:", len(ds), flush=True)

rows = []
for fn in sorted(os.listdir(SNAPS)):
    if not fn.endswith(".csv.gz"):
        continue
    df = pd.read_csv(os.path.join(SNAPS, fn), usecols=["market_id", "recorded_at"])
    df["market_id"] = df["market_id"].astype(str)
    rows.append(df)
sn = pd.concat(rows, ignore_index=True)
sn["recorded_at"] = pd.to_datetime(sn["recorded_at"], utc=True, errors="coerce")
sn = sn.dropna(subset=["recorded_at", "market_id"]).reset_index(drop=True)
print("snapsdec rows:", len(sn), "markets:", sn["market_id"].nunique(), flush=True)

# intersect markets
inter = set(sn["market_id"].unique()) & set(ds["market_id"].unique())
print("intersection markets:", len(inter), flush=True)

dsi = ds[ds["market_id"].isin(inter)].sort_values(["market_id", "decision_at"]).reset_index(drop=True)
sni = sn[sn["market_id"].isin(inter)].sort_values(["market_id", "recorded_at"]).reset_index(drop=True)
print("dsi:", len(dsi), "sni:", len(sni), flush=True)

# per-market groups; concat computed per group via sorted boundaries
def asof_pos(left_t, right_t):
    return np.searchsorted(right_t, left_t) - 1

covered = 0
lag = []
for mk in inter:
    lt = dsi.loc[dsi["market_id"] == mk, "decision_at"].values
    rt = sni.loc[sni["market_id"] == mk, "recorded_at"].values
    if len(rt) == 0:
        continue
    p = asof_pos(lt, rt)
    ok = p >= 0
    covered += int(ok.sum())
    lag.extend((lt[ok] - rt[p[ok]]).astype("timedelta64[s]").astype(float))

lag = np.asarray(lag)
print("covered:", covered, round(100.0 * covered / len(ds), 2), "%")
print("lag median sec:", float(np.median(lag)), "pct<300:", round(float((lag <= 300).mean()), 4))

# candles
cand = pd.read_csv(os.path.join(ROOT, "data", "exp", "full", "candles.csv.gz"),
                   usecols=["symbol", "interval", "close_time", "is_closed"])
cand = cand[(cand["interval"] == "15m") & (cand["is_closed"] == "t")]
cand["close_time"] = pd.to_datetime(cand["close_time"], utc=True, errors="coerce")
cand = cand.dropna(subset=["close_time"]).sort_values(["symbol", "close_time"])
a2s = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT", "DOGE": "DOGEUSDT"}
ds["_sym"] = ds["asset"].map(a2s)
cc = 0
clag = []
for sym in a2s.values():
    lt = ds.loc[ds["_sym"] == sym, "decision_at"].values
    rt = cand.loc[cand["symbol"] == sym, "close_time"].values
    p = np.searchsorted(rt, lt) - 1
    ok = p >= 0
    cc += int(ok.sum())
    clag.extend((lt[ok] - rt[p[ok]]).astype("timedelta64[s]").astype(float))
clag = np.asarray(clag)
print("candle covered:", cc, round(100.0 * cc / len(ds), 2), "%")
print("candle lag median min:", round(float(np.median(clag)) / 60, 2), "pct<35m:", round(float((clag <= 2100).mean()), 4))