"""overlap_bot.py (T08) — P0 BUY vs bot PAPER SUCCESS overlap by market.

A: bot in + P0 BUY | B: bot in + P0 SKIP | C: bot out + P0 BUY | D: neither.
Bot side: realized_pnl_usdc. P0 side: canonical net (frozen rule).
Writes OVERLAP.json to out/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical import exec_cost, settle_net

DATA = r"D:\market-policy-p0\data"
OUT = r"D:\market-policy-p0\out"
AUDIT = r"D:\lgbm-audit-v1"


def p0net(df):
    p = df["p_market_flip"].to_numpy(float)
    ask = df["candidate_ask"].to_numpy(float)
    won = (df["favorite_flip"].to_numpy(float) == 1.0)
    buy = (p - np.array([exec_cost(a) for a in ask])) > 0.03
    net = np.array([settle_net(c, w, a) if b else 0.0 for c, w, a, b in zip(
        df["candidate_side"], won, ask, buy)])
    return buy, net


def main():
    dev = pd.read_parquet(os.path.join(DATA, "OOF_FULL.parquet"),
                          columns=["opportunity_id", "market_id",
                                   "candidate_side", "candidate_ask",
                                   "p_market_flip", "favorite_flip"])
    fin = pd.read_parquet(os.path.join(OUT, "FLIP_OPPORTUNITIES_FINAL.parquet"))
    fin = fin[fin["reason_code"] == "OK"][["opportunity_id", "market_id",
                                           "candidate_side", "candidate_ask",
                                           "p_market_flip", "favorite_flip"]]
    op = pd.concat([dev, fin], ignore_index=True)
    op["market_id"] = op["market_id"].astype(str)
    t = pd.read_csv(os.path.join(AUDIT, "data", "exp", "full",
                                 "trades.csv.gz"),
                    usecols=["market_id", "mode", "status",
                             "outcome_bought", "realized_pnl_usdc",
                             "amount_usdc", "executed_price"],
                    low_memory=False, dtype={"market_id": str})
    bt = t[(t["mode"] == "PAPER") & (t["status"] == "SUCCESS")].copy()
    bt = bt.sort_values("market_id").drop_duplicates("market_id", keep="last")
    bot_mk = set(bt["market_id"])
    buy, net = p0net(op)
    op["p0_buy"] = buy
    op["p0_net"] = net
    p0_mk = set(op[op["p0_buy"]]["market_id"])
    op_mk = set(op["market_id"])
    A = bot_mk & p0_mk
    B = bot_mk - p0_mk
    C = p0_mk - bot_mk
    both_op = op_mk & bot_mk
    D = op_mk - bot_mk - p0_mk
    btp = bt.set_index("market_id")["realized_pnl_usdc"]
    netA_bot = float(btp[list(A)].sum())
    netB_bot = float(btp[list(B)].sum())
    netA_p0 = float(op[op["market_id"].isin(A)]["p0_net"].sum())
    netC_p0 = float(op[op["market_id"].isin(C)]["p0_net"].sum())
    # side agreement in A
    bside = bt.set_index("market_id")["outcome_bought"]
    agree = []
    for mk in A:
        sides = set("YES" if c == "YES" else "NO"
                    for c in op[(op["market_id"] == mk)
                                & (op["p0_buy"])]["candidate_side"])
        agree.append(str(bside.get(mk)) in sides)
    R = {"bot_markets": len(bot_mk), "bot_pnl_total": float(btp.sum()),
         "p0_buy_markets": len(p0_mk),
         "A": {"markets": len(A), "bot_pnl": netA_bot, "p0_net": netA_p0,
               "side_agree_frac": float(np.mean(agree)) if agree else None},
         "B": {"markets": len(B), "bot_pnl": netB_bot},
         "C": {"markets": len(C), "p0_net": netC_p0},
         "D": {"markets": len(D)},
         "opp_markets_with_bot": len(both_op)}
    with open(os.path.join(OUT, "OVERLAP.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("bot PAPER SUCCESS markets=%d pnl=%+.1f" % (len(bot_mk), btp.sum()))
    print("A=%d bot_pnl=%+.1f p0_net=%+.1f agree=%.3f" % (
        len(A), netA_bot, netA_p0, R["A"]["side_agree_frac"]))
    print("B=%d (bot-only) bot_pnl=%+.1f" % (len(B), netB_bot))
    print("C=%d (P0-only) p0_net=%+.1f" % (len(C), netC_p0))
    print("D=%d" % len(D))


if __name__ == "__main__":
    main()
