"""decompose_p0.py (T02) — P0 trade decomposition + equity curve data.

Reads OOF_FULL.parquet, recomputes P0 (frozen rule), breaks 548 trades down by
asset/side/window/ask-provenance/band/fold/day. Writes DECOMP.json +
EQUITY.csv to out/. Asserts slice sums == total.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import exec_cost, settle_net, price_band

DATA = r"D:\market-policy-p0\data"
OUT = r"D:\market-policy-p0\out"
MIN_EDGE = 0.03


def block(g):
    n = len(g)
    w = int(g["won"].sum())
    net = float(g["net"].sum())
    to = float(g["turnover"].sum())
    cum = np.cumsum(g.sort_values("decision_at")["net"].values)
    dd = float((np.maximum.accumulate(cum) - cum).max()) if n else 0.0
    return {"trades": n, "wins": w, "wr": w / n if n else 0.0,
            "net": net, "roi": net / to if to else 0.0, "dd": dd}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATA, "OOF_FULL.parquet"))
    p = df["p_market_flip"].to_numpy(float)
    ask = df["candidate_ask"].to_numpy(float)
    won = (df["favorite_flip"].to_numpy(float) == 1.0)
    buy = (p - np.array([exec_cost(a) for a in ask])) > MIN_EDGE
    t = df[buy].copy()
    t["won"] = won[buy]
    t["net"] = [settle_net(c, w, a) for c, w, a in
                zip(t["candidate_side"], t["won"], t["candidate_ask"])]
    t["turnover"] = 1.0
    t["band"] = [price_band(x) or "OUT_OF_BAND" for x in t["outsider_mid"]]
    t["day"] = pd.to_datetime(t["decision_at"], utc=True).dt.strftime(
        "%Y-%m-%d")
    tot = float(t["net"].sum())
    D = {"total_net": tot, "total_trades": int(len(t)), "by": {}}
    for col in ["asset", "favorite_side", "decision_window",
                "candidate_ask_provenance", "band", "fold", "day"]:
        D["by"][col] = {str(k): block(g)
                        for k, g in t.groupby(col)}
    for col in ["asset", "favorite_side", "decision_window",
                "candidate_ask_provenance", "band", "fold"]:
        s = sum(v["net"] for v in D["by"][col].values())
        assert abs(s - tot) < 1e-6, (col, s, tot)
    obs = D["by"]["candidate_ask_provenance"]
    print("net_OBSERVED=%.1f net_SYNTHETIC=%.1f" % (
        obs.get("ASK_OBSERVED", {}).get("net", 0),
        obs.get("ASK_SYNTHETIC", {}).get("net", 0)))
    eq = t.sort_values("decision_at")
    eq[["decision_at", "day", "net"]].assign(
        cumnet=np.cumsum(eq["net"].values)).to_csv(
        os.path.join(OUT, "EQUITY.csv"), index=False)
    with open(os.path.join(OUT, "DECOMP.json"), "w") as f:
        json.dump(D, f, indent=1, default=str)
    print("=== asset ===")
    for k, v in D["by"]["asset"].items():
        print(" %s n=%d wr=%.3f net=%+.1f roi=%+.3f" % (
            k, v["trades"], v["wr"], v["net"], v["roi"]))
    print("=== side ===")
    for k, v in D["by"]["favorite_side"].items():
        print(" fav_%s n=%d wr=%.3f net=%+.1f" % (k, v["trades"], v["wr"],
                                                  v["net"]))
    print("=== window ===")
    for k, v in D["by"]["decision_window"].items():
        print(" %s n=%d wr=%.3f net=%+.1f" % (k, v["trades"], v["wr"],
                                              v["net"]))
    print("=== band ===")
    for k, v in D["by"]["band"].items():
        print(" %s n=%d wr=%.3f net=%+.1f" % (k, v["trades"], v["wr"],
                                              v["net"]))
    print("=== fold ===")
    for k, v in D["by"]["fold"].items():
        print(" %s n=%d wr=%.3f net=%+.1f" % (k, v["trades"], v["wr"],
                                              v["net"]))
    topd = max(D["by"]["day"].items(), key=lambda kv: kv[1]["net"])
    print("top day: %s net=%+.1f share=%.3f" % (
        topd[0], topd[1]["net"], topd[1]["net"] / tot))
    conc = [k for k, v in list(D["by"]["asset"].items())
            + list(D["by"]["day"].items()) if v["net"] / tot > 0.6]
    print("CONCENTRATION_RISK slices (>60%%):", conc if conc else "none")


if __name__ == "__main__":
    main()
