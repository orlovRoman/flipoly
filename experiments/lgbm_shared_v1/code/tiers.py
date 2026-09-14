"""tiers.py v1.0.0 — settings Tier A/B/C per decision row (spec item 11).

Tier A (probative replay): model mapped + registry thresholds + quotes OK +
  era known + min_edge present + MRF present iff decision >= 2026-08-21
  (MRF telemetry starts 08-21) + max_acceptable_price present iff BUY
  (BUY-path guard value; SKIP rows never reach it).
Tier B (fixed scenarios only): mapped + quotes OK, rest partial.
Tier C (insufficient): NO_MODEL / UNKNOWN_MODEL / unmapped / no quotes /
  era unknown.
"""
import numpy as np

MRF_FROM = "2026-08-21"


def _present(v):
    import pandas as pd
    if v.dtype == object:
        return pd.notna(v) & (v != "") & (v != "None")
    return pd.notna(v)


def assign_tiers(df, mapped, thresholds_ok):
    n = len(df)
    tier = np.full(n, "C", dtype=object)
    reasons = np.full(n, "", dtype=object)
    qok = df["quote_ok"].fillna(False).to_numpy()
    is_model = ~df["model_slot"].isin(["NO_MODEL", "UNKNOWN_MODEL"]).to_numpy()
    me = _present(df["min_edge_used"]) if "min_edge_used" in df.columns else np.zeros(n, dtype=bool)
    mrf = _present(df["mrf_mode"]) if "mrf_mode" in df.columns else np.zeros(n, dtype=bool)
    day = df["decision_at"].dt.strftime("%Y-%m-%d").to_numpy()
    mrf_need = day >= MRF_FROM
    mrf_ok = mrf | ~mrf_need
    fa = df["final_action"].fillna("").to_numpy()
    mx = _present(df["max_acceptable_price"]) if "max_acceptable_price" in df.columns else np.zeros(n, dtype=bool)
    mx_ok = mx | (~np.isin(fa, ["BUY_YES", "BUY_NO"]))
    a = mapped & thresholds_ok & qok & me & mrf_ok & mx_ok & is_model
    b = (~a) & mapped & qok & is_model
    tier[a] = "A"
    tier[b] = "B"
    reasons[~mapped & is_model] = "model_unmapped;"
    reasons[~qok] = reasons[~qok] + "no_quotes;"
    reasons[df["model_slot"].to_numpy() == "NO_MODEL"] = \
        reasons[df["model_slot"].to_numpy() == "NO_MODEL"] + "no_model;"
    reasons[df["model_slot"].to_numpy() == "UNKNOWN_MODEL"] = \
        reasons[df["model_slot"].to_numpy() == "UNKNOWN_MODEL"] + "unknown_model;"
    return tier, reasons
