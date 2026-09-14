"""quotes.py v1.0.0 — causal L1 snapshot per decision (spec steps 31-33).

Match: last snapshot with recorded_at <= decision_at AND received <= decision_at
per market (merge_asof primary + fallback for received violations).
YES/NO asks with synthetic_NO flag; validity flags; mid with validity.
"""
import numpy as np
import pandas as pd


def _valid(s):
    s = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    return np.where((s > 0) & (s < 1), s, np.nan)


def attach_quotes(ev, snaps):
    """snaps columns include recorded_at/received_timestamp + book fields.
    Per-market merge_asof (obviously correct ordering) + received guard with
    fallback + vectorized post-assert that matched recorded_at <= decision_at.
    """
    keep = ["market_id", "recorded_at", "received_timestamp", "mid_price",
            "best_bid", "best_ask", "poly_up_best_bid", "poly_up_best_ask",
            "poly_up_mid", "poly_down_best_bid", "poly_down_best_ask",
            "poly_down_mid", "spread", "time_left_seconds", "market_end_at",
            "final_outcome", "strike_value", "strike_source",
            "binance_price", "oracle_price"]
    s = snaps[keep].copy()
    if len(ev) == 0:
        out = ev.copy()
        for c in ["mid_price", "best_bid", "best_ask", "poly_up_best_ask",
                  "poly_down_best_ask", "spread", "time_left_seconds",
                  "final_outcome"]:
            out[c] = np.nan
        out["yes_ask"] = np.nan
        out["no_ask"] = np.nan
        out["synthetic_no"] = False
        out["quote_ok"] = False
        out["p_market_yes"] = np.nan
        out["arb_flag"] = False
        return out, {"matched": 0, "received_violations": 0,
                     "fallback_applied": 0, "quote_ok_rate": 0.0,
                     "synthetic_rate": 0.0}
    parts = []
    stats = {"received_violations": 0, "fallback_applied": 0}
    for mk, g in ev.groupby("market_id", sort=True):
        sg = s[s["market_id"] == mk].sort_values(
            "recorded_at", kind="mergesort").reset_index(drop=True)
        eg = g.sort_values("decision_at", kind="mergesort").reset_index(drop=True)
        if len(sg) == 0:
            m = eg.copy()
            for c in keep:
                if c != "market_id":
                    m[c] = np.nan
            parts.append(m)
            continue
        left = eg[["decision_event_id", "decision_at"]].copy()
        right = sg.rename(columns={"recorded_at": "_r"})
        m = pd.merge_asof(left.sort_values("decision_at"),
                          right.sort_values("_r"),
                          left_on="decision_at", right_on="_r",
                          direction="backward", allow_exact_matches=True)
        bad = m["received_timestamp"].notna() & (m["received_timestamp"] > m["decision_at"])
        stats["received_violations"] += int(bad.sum())
        if bad.any():
            rec = sg["recorded_at"].to_numpy()
            rvd = sg["received_timestamp"].to_numpy()
            for i in m.index[bad].tolist():
                dec = m.at[i, "decision_at"]
                cand = np.nonzero((rec <= dec) & (rvd <= dec))[0]
                if len(cand):
                    j = cand[-1]
                    m.at[i, "_r"] = sg.iloc[j]["recorded_at"]
                    for c in keep:
                        if c != "market_id":
                            m.at[i, c] = sg.iloc[j][c]
                else:
                    m.at[i, "_r"] = pd.NaT
                    for c in keep:
                        if c != "market_id":
                            m.at[i, c] = np.nan
            stats["fallback_applied"] += int(bad.sum())
        m["market_id"] = mk
        parts.append(m)
    m = pd.concat(parts, ignore_index=True)
    got = m["mid_price"].notna()
    assert bool(((m.loc[got, "_r"] <= m.loc[got, "decision_at"])).all()), \
        "causality breach in quote match"
    stats["matched"] = int(got.sum())
    out = ev.merge(m[["decision_event_id", "mid_price", "best_bid", "best_ask",
                      "poly_up_best_bid", "poly_up_best_ask", "poly_up_mid",
                      "poly_down_best_bid", "poly_down_best_ask", "poly_down_mid",
                      "spread", "time_left_seconds", "market_end_at",
                      "final_outcome", "strike_value", "strike_source",
                      "binance_price", "oracle_price"]],
                   on="decision_event_id", how="left")
    up_ask = _valid(out["poly_up_best_ask"].where(
        out["poly_up_best_ask"].notna(), out["best_ask"]))
    dn_raw = out["poly_down_best_ask"].to_numpy()
    synth = ~(pd.to_numeric(dn_raw, errors="coerce") > 0)
    bb = pd.to_numeric(out["best_bid"], errors="coerce").to_numpy(dtype=np.float64)
    dn_fill = np.where((bb > 0) & (bb < 1), 1.0 - bb, np.nan)
    dn = _valid(pd.Series(np.where(synth, dn_fill,
                                   pd.to_numeric(dn_raw, errors="coerce"))))
    out["yes_ask"] = up_ask
    out["no_ask"] = dn
    out["synthetic_no"] = synth & out["no_ask"].notna().to_numpy()
    out["quote_ok"] = out["yes_ask"].notna() & out["no_ask"].notna()
    out["p_market_yes"] = _valid(out["poly_up_mid"].where(
        out["poly_up_mid"].notna(), out["mid_price"]))
    out["arb_flag"] = ((out["yes_ask"] + out["no_ask"]) < 1.0).fillna(False).to_numpy()
    stats["quote_ok_rate"] = float(out["quote_ok"].mean()) if len(out) else 0.0
    stats["synthetic_rate"] = float(out["synthetic_no"].mean()) if len(out) else 0.0
    return out, stats
