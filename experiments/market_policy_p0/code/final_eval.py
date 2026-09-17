"""final_eval.py (T04-T07) — P0 on FINAL + threshold sweep + bootstrap/Sharpe.

Frozen P0 rule (min_edge=0.03). Sweep is DIAGNOSTIC (threshold stays frozen).
Bootstrap by day and market, 2000 reps, seed 20260913.
Writes FINAL_P0.json, SWEEP.csv, BOOT.json to out/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import exec_cost, settle_net

OUT = r"D:\market-policy-p0\out"
DATA = r"D:\market-policy-p0\data"
SEED = 20260913
REPS = 2000
SWEEP = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]


def apply_p0(df, min_edge):
    p = df["p_market_flip"].to_numpy(float)
    ask = df["candidate_ask"].to_numpy(float)
    won = (df["favorite_flip"].to_numpy(float) == 1.0)
    buy = (p - np.array([exec_cost(a) for a in ask])) > min_edge
    net = np.array([settle_net(c, w, a) if b else 0.0 for c, w, a, b in zip(
        df["candidate_side"], won, ask, buy)])
    return buy, net


def summ_net(df, net, name):
    df = df.copy()
    df["day"] = pd.to_datetime(df["decision_at"], utc=True).dt.strftime(
        "%Y-%m-%d")
    tot = float(net.sum())
    by_day = pd.Series(net).groupby(df["day"].values).sum()
    by_mkt = pd.Series(net).groupby(df["market_id"].values).sum()
    cum = np.cumsum(net[np.argsort(
        pd.to_datetime(df["decision_at"], utc=True).values)])
    dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    dn = by_day.values
    sharpe = (float(dn.mean() / dn.std() * np.sqrt(365))
              if len(dn) > 2 and dn.std() > 0 else None)
    return {"name": name, "n": int(len(df)),
            "trades": int((net != 0).sum()), "net": tot,
            "roi": float(tot / (net != 0).sum()) if (net != 0).any() else 0.0,
            "dd": dd,
            "top1day": float(by_day.max() / tot) if tot > 0 else None,
            "top1mkt": float(by_mkt.max() / tot) if tot > 0 else None,
            "sharpe_daily": sharpe,
            "wr": float(((net > 0).sum()) / (net != 0).sum())
            if (net != 0).any() else 0.0}


def boot_ci(net, groups, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    uk = np.unique(groups)
    idx = {u: np.nonzero(groups == u)[0] for u in uk}
    boots = np.empty(reps)
    for r in range(reps):
        pick = uk[rng.integers(0, len(uk), len(uk))]
        boots[r] = net[np.concatenate([idx[u] for u in pick])].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi),
            "p_pos": float((boots > 0).mean())}


def main():
    fin = pd.read_parquet(os.path.join(OUT, "FLIP_OPPORTUNITIES_FINAL.parquet"))
    fin = fin[fin["reason_code"] == "OK"].copy()
    dev = pd.read_parquet(os.path.join(DATA, "OOF_FULL.parquet"))
    # T04: frozen P0 on FINAL
    buy, net = apply_p0(fin, 0.03)
    F = summ_net(fin, net, "FINAL")
    F["by_day"] = boot_ci(net, fin.assign(
        day=pd.to_datetime(fin["decision_at"], utc=True).dt.strftime(
            "%Y-%m-%d"))["day"].to_numpy())
    F["by_market"] = boot_ci(net, fin["market_id"].to_numpy())
    print("FINAL: n=%d trades=%d net=%+.1f wr=%.3f dd=%.1f sharpe=%s" % (
        F["n"], F["trades"], F["net"], F["wr"], F["dd"], F["sharpe_daily"]))
    print("  CI_day=[%.1f, %.1f] p_pos=%.3f" % (F["by_day"]["lo"],
                                                F["by_day"]["hi"],
                                                F["by_day"]["p_pos"]))
    print("  CI_mkt=[%.1f, %.1f] p_pos=%.3f" % (F["by_market"]["lo"],
                                                F["by_market"]["hi"],
                                                F["by_market"]["p_pos"]))
    # OBSERVED vs SYNTHETIC on FINAL
    for prov in ["ASK_OBSERVED", "ASK_SYNTHETIC"]:
        g = fin[fin["candidate_ask_provenance"] == prov]
        _, ng = apply_p0(g, 0.03)
        s = summ_net(g, ng, "FINAL_" + prov)
        F["split_" + prov] = {"trades": s["trades"], "net": s["net"],
                              "wr": s["wr"]}
        print("  %s: trades=%d net=%+.1f wr=%.3f" % (prov, s["trades"],
                                                     s["net"], s["wr"]))
    with open(os.path.join(OUT, "FINAL_P0.json"), "w") as f:
        json.dump(F, f, indent=1, default=str)
    # T05: sweep on dev + final combined (diagnostic)
    both = pd.concat([dev[["p_market_flip", "candidate_ask",
                           "favorite_flip", "candidate_side"]],
                      fin[["p_market_flip", "candidate_ask",
                           "favorite_flip", "candidate_side"]]],
                     ignore_index=True)
    rows = []
    for me in SWEEP:
        _, n2 = apply_p0(both, me)
        rows.append({"min_edge": me, "trades": int((n2 != 0).sum()),
                     "net": float(n2.sum())})
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(OUT, "SWEEP.csv"), index=False)
    print("=== sweep (dev+final) ===")
    for _, r in sw.iterrows():
        print(" edge=%.2f trades=%d net=%+.1f" % (r["min_edge"], r["trades"],
                                                  r["net"]))
    pos = sw[sw["net"] > 0]["min_edge"]
    width = (float(pos.max() - pos.min()) if len(pos) > 1 else 0.0)
    print("plateau width (net>0): %.2f" % width)
    # T06: bootstrap on dev (for the record) + combined
    B = {}
    for name, frame in (("dev", dev), ("final", fin)):
        if name == "dev":
            _, nd = apply_p0(frame, 0.03)
            dd = frame.assign(day=pd.to_datetime(
                frame["decision_at"], utc=True).dt.strftime("%Y-%m-%d"))
        else:
            nd = net
            dd = frame.assign(day=pd.to_datetime(
                frame["decision_at"], utc=True).dt.strftime("%Y-%m-%d"))
        B[name] = {"by_day": boot_ci(nd, dd["day"].to_numpy()),
                   "by_market": boot_ci(nd, dd["market_id"].to_numpy()
                                        if "market_id" in dd.columns
                                        else dd["day"].to_numpy())}
    with open(os.path.join(OUT, "BOOT.json"), "w") as f:
        json.dump({"final": F, "sweep_width": width, "boot": B}, f, indent=1,
                  default=str)
    print("wrote FINAL_P0.json SWEEP.csv BOOT.json")


if __name__ == "__main__":
    main()
