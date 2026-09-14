"""Slice test 2: targets + tiers + execution on 08-03. Must pass before full build."""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\lgbm-audit-v1\code\shared")
import load
import decisions
import quotes
import targets
import tiers
import execution

t0 = time.time()
f = load.read_funnel_day("2026-08-03")
s = load.read_snaps_day("2026-08-03")
ev = decisions.build_events(f)
out, qs = quotes.attach_quotes(ev, s)
print("base rows=%d (%.1fs)" % (len(out), time.time() - t0))

out["contract_target"] = np.where(
    out["final_outcome"] == "YES", 1.0,
    np.where(out["final_outcome"] == "NO", 0.0, np.nan))
lm = load.read_livemarkets()
lm["end3"] = pd.to_datetime(lm["market_end_at"], utc=True, errors="coerce")
m = lm["end3"].isna()
lm.loc[m, "end3"] = pd.to_datetime(lm.loc[m, "end_time_est"], utc=True, errors="coerce")
ends = dict(zip(lm["market_id"].astype(str), lm["end3"]))
out["market_end"] = out["market_id"].map(ends)
print("ends resolved: %.3f" % out["market_end"].notna().mean())

candles = load.read_candles()
candles = candles[candles["open_time"].notna()].reset_index(drop=True)
for iv, add in (("5m", 299.999), ("15m", 899.999)):
    mm = candles["close_time"].isna() & (candles["interval"] == iv)
    candles.loc[mm, "close_time"] = (
        candles.loc[mm, "open_time"] + pd.to_timedelta(add, unit="s"))
carr = targets.build_candle_arrays(candles)
leg, lend = targets.legacy_native_for(out, carr)
print("legacy_native hit rate: %.3f" % pd.Series(leg).notna().mean())

inv = load.read_inventory()
inv["key"] = inv["asset"].astype(str) + "|" + inv["version"].astype(str)
reg = inv.set_index("key")
key = out["model_slot"].astype(str) + "|" + out["slot_version"].apply(
    lambda v: str(int(float(v))) if str(v) not in ("nan", "None", "") else "nan")
hit = (~out["model_slot"].isin(["NO_MODEL", "UNKNOWN_MODEL"])) & key.isin(reg.index)
print("mapped: %d / %d" % (int(hit.sum()), len(out)))

trades = load.read_trades()
AF = execution.agg_frame(execution.precompute(
    trades, load.read_exec("exrequests"), load.read_exec("exattempts"),
    load.read_exec("exfills")))
print("run aggregates:", len(AF))
r0 = execution.apply_runs(out, AF)
print("r0 with fills: %d rows, net sum=%+.2f" % (
    int((np.asarray(r0[4]) > 0).sum()), float(np.nansum(r0[0]))))
print("SLICE2-OK")
