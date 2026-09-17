"""economics.py — canonical policies P0..P5, nets, concentration (spec v1.1.0).

Rule (DEVELOPMENT fixed): BUY iff p_flip - all_in_cost > 0.03 else SKIP.
Reads OOF_FULL.parquet. Writes POLICY_NET.csv + ECON_SUMMARY.json to runs/.
Bootstrap paired net deltas by day and by market.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import exec_cost, settle_net

RUNDIR = r"D:\lgbm-favorite-flip-v1\runs"
SEED = 20260913
REPS = 2000
MIN_EDGE = 0.03
POLICIES = [("P0_market", "p_market_flip"), ("P1_logreg", "p_logreg_flip"),
            ("P2_L0", "cal_L0"), ("P3_L1", "cal_L1"), ("P4_L2", "cal_L2"),
            ("P5_L3", "cal_L3")]


def boot_delta_vals(d, groups, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    uk = np.unique(groups)
    idx = {u: np.nonzero(groups == u)[0] for u in uk}
    est = float(d.mean())
    boots = np.empty(reps)
    for r in range(reps):
        pick = uk[rng.integers(0, len(uk), len(uk))]
        sel = np.concatenate([idx[u] for u in pick])
        boots[r] = d[sel].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi)}


def main():
    df = pd.read_parquet(os.path.join(RUNDIR, "OOF_FULL.parquet"))
    df["day"] = pd.to_datetime(df["decision_at"], utc=True).dt.strftime(
        "%Y-%m-%d")
    eco = df[df["econ_available"]].copy().sort_values("decision_at")
    eco["won"] = (eco["favorite_flip"] == 1).to_numpy()
    out = eco[["opportunity_id", "market_id", "day", "asset",
               "favorite_side", "decision_window", "candidate_side",
               "candidate_ask", "favorite_flip",
               "candidate_ask_provenance"]].copy()
    summ = {}
    nets = {}
    for pname, pcol in POLICIES:
        sub = eco if pcol != "p_logreg_flip" else eco[eco["logreg_available"]]
        p = sub[pcol].to_numpy(float)
        ask = sub["candidate_ask"].to_numpy(float)
        edge = p - np.array([exec_cost(a) for a in ask])
        buy = edge > MIN_EDGE
        net = np.array([settle_net(c, w, a) if b else 0.0
                        for c, w, a, b in zip(sub["candidate_side"], sub["won"],
                                              ask, buy)])
        key = "net_" + pname
        out[key] = np.nan
        out.loc[sub.index, key] = net
        cum = np.cumsum(net)
        dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
        by_day = pd.Series(net).groupby(sub["day"].values).sum()
        by_mkt = pd.Series(net).groupby(sub["market_id"].values).sum()
        tot = float(net.sum())
        summ[pname] = {
            "opportunities": int(len(sub)), "trades": int(buy.sum()),
            "coverage": float(buy.mean()) if len(sub) else 0.0,
            "turnover": int(buy.sum()), "net": tot,
            "roi": float(tot / buy.sum()) if buy.sum() else 0.0,
            "max_drawdown": dd,
            "top1_day_share": (float(by_day.max() / tot)
                               if tot > 0 and len(by_day) else None),
            "top5_day_share": (float(by_day.nlargest(5).sum() / tot)
                               if tot > 0 and len(by_day) else None),
            "top1_mkt_share": (float(by_mkt.max() / tot)
                               if tot > 0 and len(by_mkt) else None),
            "net_by_day": {k: float(v) for k, v in by_day.items()}}
        nets[pname] = pd.Series(net, index=sub.index)
        # observed vs synthetic ask split
        sub = sub.copy()
        sub["_buy"] = buy
        sub["_net"] = net
        for prov, g in sub.groupby("candidate_ask_provenance"):
            summ[pname]["net_" + prov] = float(g["_net"].sum())
            summ[pname]["trades_" + prov] = int(g["_buy"].sum())
    # paired net deltas (common IDs)
    comp = {}
    base = {"market": "net_P0_market", "logreg": "net_P1_logreg"}
    for m, mp in [("L0", "net_P2_L0"), ("L1", "net_P3_L1"),
                  ("L2", "net_P4_L2"), ("L3", "net_P5_L3")]:
        for c, cp in base.items():
            a = out[mp].to_numpy(float)
            b = out[cp].to_numpy(float)
            okm = np.isfinite(a) & np.isfinite(b)
            d = a[okm] - b[okm]
            g = out.loc[okm]
            comp["%s_minus_%s" % (mp, cp)] = {
                "n": int(okm.sum()),
                "by_day": boot_delta_vals(d, g["day"].to_numpy()),
                "by_market": boot_delta_vals(d, g["market_id"].to_numpy())}
    summ["paired_net"] = comp
    out.to_csv(os.path.join(RUNDIR, "POLICY_NET.csv"), index=False)
    with open(os.path.join(RUNDIR, "ECON_SUMMARY.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    print("=== policies ===")
    for pname, _ in POLICIES:
        s = summ[pname]
        print("%s opp=%d trades=%d cov=%.3f net=%+.3f roi=%+.4f dd=%.3f" % (
            pname, s["opportunities"], s["trades"], s["coverage"], s["net"],
            s["roi"], s["max_drawdown"]))
    print("=== paired net increment by_market ===")
    for k, v in comp.items():
        b = v["by_market"]
        print("%s n=%d: %+.4f [%+.4f, %+.4f]" % (k, v["n"], b["est"],
                                                 b["lo"], b["hi"]))


if __name__ == "__main__":
    main()
