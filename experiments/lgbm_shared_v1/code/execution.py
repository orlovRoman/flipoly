"""execution.py v1.0.0 — trade/fill joins + R0 actuals (spec item 13, E6).

Precompute per-run fill aggregates once (small tables), then map per decision.
Payout needs the decision's market outcome (won_yes or NaN when unresolved).
"""
import numpy as np
import pandas as pd


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def precompute(trades, reqs, atts, fills):
    """Returns dict run_id -> aggregate + trades-without-run count info."""
    t = trades[["id", "decision_run_id"]].copy()
    t["id"] = t["id"].astype(str)
    t["decision_run_id"] = t["decision_run_id"].astype(str)
    r = reqs[["id", "trade_history_id", "outcome_to_buy"]].copy()
    r["id"] = r["id"].astype(str)
    r["trade_history_id"] = r["trade_history_id"].astype(str)
    a = atts[["id", "request_id"]].copy()
    a["id"] = a["id"].astype(str)
    a["request_id"] = a["request_id"].astype(str)
    f = fills[["attempt_id", "price", "shares", "fee_usdc"]].copy()
    f["attempt_id"] = f["attempt_id"].astype(str)
    for c in ("price", "shares", "fee_usdc"):
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0).to_numpy()
    m = f.merge(a, left_on="attempt_id", right_on="id", how="left",
                suffixes=("", "_a"))
    m = m.merge(r, left_on="request_id", right_on="id", how="left",
                suffixes=("", "_r"))
    m = m.merge(t, left_on="trade_history_id", right_on="id", how="left",
                suffixes=("", "_t"))
    m["cost"] = m["shares"] * m["price"]
    agg = {}
    for run, g in m.groupby("decision_run_id", sort=True):
        if run in ("", "None", "nan") or pd.isna(run):
            continue
        sides = g["outcome_to_buy"].fillna("").unique().tolist()
        bad = [s for s in sides if s not in ("YES", "NO", "")]
        agg[run] = {
            "sh_yes": float(g.loc[g["outcome_to_buy"] == "YES", "shares"].sum()),
            "sh_no": float(g.loc[g["outcome_to_buy"] == "NO", "shares"].sum()),
            "cost": float(g["cost"].sum()),
            "fee": float(g["fee_usdc"].sum()),
            "n_fills": int(len(g)),
            "n_trades": int(g["trade_history_id"].nunique()),
            "side_flag": ("SIDE_" + ",".join(sorted(bad))) if bad else "",
        }
    return agg


def agg_frame(agg):
    AF = pd.DataFrame.from_dict(agg, orient="index")
    AF.index.name = "run_id"
    return AF.reset_index()


def apply_runs(df, AF):
    """Vectorized R0 per decision row via merge. Returns
    (net, cost, fee, n_trades, n_fills, flag) arrays."""
    m = df[["decision_run_id"]].copy()
    m["decision_run_id"] = m["decision_run_id"].astype(str)
    j = m.merge(AF, left_on="decision_run_id", right_on="run_id", how="left")
    won = df["contract_target"].to_numpy(dtype=np.float64)
    has = j["run_id"].notna().to_numpy()
    sh_y = j["sh_yes"].fillna(0.0).to_numpy(dtype=np.float64)
    sh_n = j["sh_no"].fillna(0.0).to_numpy(dtype=np.float64)
    cost = j["cost"].fillna(0.0).to_numpy(dtype=np.float64)
    fee = j["fee"].fillna(0.0).to_numpy(dtype=np.float64)
    py = np.where(won == 1.0, 1.0, 0.0)
    pn = 1.0 - py
    net = np.where(has & np.isfinite(won), sh_y * py + sh_n * pn - cost - fee,
                   np.where(has, np.nan, 0.0))
    flag = np.where(has, j["side_flag"].fillna("").to_numpy(), "NO_RUN")
    norun = (df["decision_run_id"].astype(str).isin(["", "None", "nan"])).to_numpy()
    flag = np.where(~has & norun, "NO_RUN_ID", flag)
    flag = np.where(has & ~np.isfinite(won), "NO_OUTCOME", flag)
    nt = j["n_trades"].fillna(0).to_numpy(dtype=np.int64)
    nf = j["n_fills"].fillna(0).to_numpy(dtype=np.int64)
    return net, cost, fee, nt, nf, flag
