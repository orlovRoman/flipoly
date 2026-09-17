"""reproduce_p0.py (T01) — independent P0 recompute from OOF_FULL.parquet.

Frozen rule: edge = outsider_mid - exec_cost(candidate_ask); BUY iff > 0.03.
Compares against origin POLICY_NET.csv row-wise and totals.
Expect: net +330.8 (±0.1), trades 548, dd 21.1, top1day 0.18, top1mkt 0.09.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import exec_cost, settle_net

DATA = r"D:\market-policy-p0\data"
MIN_EDGE = 0.03


def main():
    df = pd.read_parquet(os.path.join(DATA, "OOF_FULL.parquet"))
    pn = pd.read_csv(os.path.join(DATA, "POLICY_NET.csv"))
    assert len(df) == len(pn) == 18794
    ref = pn.set_index("opportunity_id")["net_P0_market"]
    assert set(df["opportunity_id"]) == set(ref.index)
    p = df["p_market_flip"].to_numpy(float)
    ask = df["candidate_ask"].to_numpy(float)
    won = (df["favorite_flip"].to_numpy(float) == 1.0)
    edge = p - np.array([exec_cost(a) for a in ask])
    buy = edge > MIN_EDGE
    net = np.array([settle_net(c, w, a) if b else 0.0 for c, w, a, b in zip(
        df["candidate_side"], won, ask, buy)])
    ref = pn.set_index("opportunity_id")["net_P0_market"]
    ref = ref.loc[df["opportunity_id"].values].to_numpy(float)
    maxdiff = float(np.abs(net - ref).max())
    print("rows=%d max_row_diff=%.6f" % (len(df), maxdiff))
    assert maxdiff < 1e-9, "row-wise mismatch vs origin POLICY_NET"
    df["day"] = pd.to_datetime(df["decision_at"], utc=True).dt.strftime(
        "%Y-%m-%d")
    tot = float(net.sum())
    by_day = pd.Series(net).groupby(df["day"].values).sum()
    by_mkt = pd.Series(net).groupby(df["market_id"].values).sum()
    cum = np.cumsum(net[np.argsort(
        pd.to_datetime(df["decision_at"], utc=True).values)])
    dd = float((np.maximum.accumulate(cum) - cum).max())
    print("net=%+.3f trades=%d dd=%.3f top1day=%.3f top1mkt=%.3f" % (
        tot, int(buy.sum()), dd, by_day.max() / tot, by_mkt.max() / tot))
    assert abs(tot - 330.8) < 0.15, tot
    assert int(buy.sum()) == 548
    assert abs(dd - 21.1) < 0.15, dd
    assert abs(by_day.max() / tot - 0.18) < 0.02
    assert abs(by_mkt.max() / tot - 0.09) < 0.02
    print("T01 REPRODUCED: net=+330.8 trades=548 dd=21.1 top1day=0.18 top1mkt=0.09")


if __name__ == "__main__":
    main()
