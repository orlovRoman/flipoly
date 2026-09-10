"""Legacy LogReg comparison engine (LOCAL ONLY, chunked).

Opportunities: one row per resolved market at the fixed causal decision
(first snapshot in (T-5m, T-4:45m]); features [mid_price, spread,
time_left_min derived]; flip label from market outcome; phase slice from
|mid-0.5| at decision (contested/leaning/decided bands).

Scored here: the 3-feature family (v11 + successor + 12 peers) via pinned
artifacts. v22 (26-feature pipeline) is PENDING_FEATURE_MAP — never scored
on a feature subset silently.

Common fixed policy (old rules not reconstructed as fact): outsider side by
lower mid; enter if mapped p_win >= 0.55 and top ask in [0.01, 0.40];
budget $1 purchase cost; snapshot top-ask fill (TOP_ONLY_ASSUMED, no depth
pre-09-09); fee UNKNOWN -> gross + signed scenarios.
"""
from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from polyflip.research.legacy_btc_logreg.pinned_loader import (
    flip_to_side_probs, load_pinned,
)

FEATURES_3 = ["mid_price", "spread", "time_left_min"]
# Fixed common gate = OBSERVED-effective old rule (not invented): global
# FLIP_THRESHOLD 0.2 (uniform across leaning models in funnel 08-20..09-09)
# + abstain band 0.05 (config default) -> enter iff p_flip >= 0.25.
THR = 0.25
ASK_MIN, ASK_MAX = 0.01, 0.40
BUDGET = 1.0
FEE_SCENARIOS = (0.0, 0.001, 0.002)


def phase_of(dev: float) -> str:
    if dev < 0.10:
        return "contested"
    if dev < 0.25:
        return "leaning"
    return "decided"


def build_opportunities(snaps: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """One decision row per market (resolved YES/NO only)."""
    snaps = snaps.copy()
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"], utc=True, format="mixed")
    snaps["market_id"] = snaps["market_id"].astype(str)
    live = live.copy().set_index(live["market_id"].astype(str))
    out = []
    for mid, g in snaps.groupby("market_id"):
        if mid not in live.index:
            continue
        end = pd.to_datetime(live.loc[mid, "end_time_est"], utc=True)
        if pd.isna(end):
            continue
        tgt = end - timedelta(seconds=300)
        cand = g[(g["recorded_at"] > tgt)
                 & (g["recorded_at"] <= tgt + timedelta(seconds=15))]
        cand = cand.dropna(subset=["mid_price", "best_ask", "spread"])
        if cand.empty:
            continue
        dec = cand.sort_values("recorded_at").iloc[0]
        outcome = dec["final_outcome"]
        if outcome not in ("YES", "NO"):
            continue
        mid_px = float(dec["mid_price"])
        if mid_px == 0.5:
            continue
        flip = int((mid_px > 0.5) != (outcome == "YES"))
        tlm = (end - dec["recorded_at"]).total_seconds() / 60.0
        out.append({
            "market_id": mid, "asset": live.loc[mid, "asset"],
            "decision_at": dec["recorded_at"], "end_at": end,
            "mid_price": mid_px, "spread": float(dec["spread"]),
            "time_left_min": tlm, "ask": float(dec["best_ask"]),
            "target_flip": flip, "target_up": 1 if outcome == "YES" else 0,
            "phase": phase_of(abs(mid_px - 0.5)),
        })
    df = pd.DataFrame(out)
    assert df["market_id"].is_unique, "one row per market"
    return df


def score_models(opps: pd.DataFrame, artifacts: dict[int, tuple[bytes, str]]) -> pd.DataFrame:
    """Add p_flip_<id> columns. artifacts: registry_id -> (blob, sha)."""
    df = opps.copy()
    rows = df[FEATURES_3].to_dict("records")
    for rid, (blob, sha) in sorted(artifacts.items()):
        model, order = load_pinned(blob, sha, FEATURES_3)
        coef = model.coef_[0]
        b = float(model.intercept_[0])
        ps = []
        for r in rows:
            z = b + sum(c * r[f] for c, f in zip(coef, order))
            ps.append(1.0 / (1.0 + math.exp(-z)))
        df[f"p_flip_{rid}"] = ps
    return df


def brier(ps, ys) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(len(ps), 1)


def logloss(ps, ys, eps=1e-6) -> float:
    s = 0.0
    for p, y in zip(ps, ys):
        p = min(max(p, eps), 1 - eps)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / max(len(ps), 1)


def forecast_table(df: pd.DataFrame, ids: list[int]) -> dict:
    """Brier/logloss overall + by side/price/phase/week on common rows."""
    ys = df["target_flip"].tolist()
    sides = (df["mid_price"] > 0.5).map({True: "UPfav", False: "DOWNfav"})
    out = {}
    for rid in ids:
        ps = df[f"p_flip_{rid}"].tolist()
        rec = {"n": len(ps), "brier": round(brier(ps, ys), 5),
               "logloss": round(logloss(ps, ys), 5)}
        rec["by_side"] = {s: {"n": int((sides == s).sum()),
                              "brier": round(brier([p for p, q in zip(ps, sides) if q == s],
                                                   [y for y, q in zip(ys, sides) if q == s]), 5)}
                          for s in ("UPfav", "DOWNfav")}
        rec["by_phase"] = {p: {"n": int((df["phase"] == p).sum()),
                               "brier": round(brier(df.loc[df["phase"] == p, f"p_flip_{rid}"],
                                                    df.loc[df["phase"] == p, "target_flip"]), 5)}
                           for p in sorted(df["phase"].unique())}
        out[str(rid)] = rec
    return out


def apply_policy(df: pd.DataFrame, rid: int) -> pd.DataFrame:
    """Fixed common outsider policy for one model's p_flip. Returns fills."""
    d = df.copy()
    d["p_flip"] = d[f"p_flip_{rid}"]
    d["outsider"] = d["mid_price"].map(lambda m: "DOWN" if m > 0.5 else "UP")
    mapped = [flip_to_side_probs(m, p) for m, p in zip(d["mid_price"], d["p_flip"])]
    d["p_win"] = [m["p_down"] if s == "DOWN" else m["p_up"]
                  for m, s in zip(mapped, d["outsider"])]
    # Mirrored observed old gates: price + spread/mid<=0.08 (funnel top skips).
    # Time gate [60,600]s holds by construction (fixed T-5m = 300 s).
    # MRF regime vetoes are unmirrored (separate model family; disclosed).
    d["enter"] = (d["p_win"] >= THR) & d["ask"].between(ASK_MIN, ASK_MAX) & \
                 ((d["spread"] / d["mid_price"]) <= 0.08)
    d["shares"] = (BUDGET / d["ask"]).where(d["enter"], 0.0)
    won = ((d["outsider"] == "UP") & (d["target_up"] == 1)) | \
          ((d["outsider"] == "DOWN") & (d["target_up"] == 0))
    d["payout"] = (d["shares"] * won).astype(float)
    d["spent"] = (d["shares"] * d["ask"]).fillna(0.0)
    d["gross"] = (d["payout"] - d["spent"]).where(d["enter"], 0.0)
    for f in FEE_SCENARIOS:
        d[f"fee_{f}"] = (d["gross"] - d["spent"] * f).where(d["enter"], 0.0)
    return d


def decompose(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """Decision-overlap groups for two scored policy frames (same index)."""
    ea, eb = a["enter"], b["enter"]
    groups = {
        "both_enter": ea & eb, "only_a": ea & ~eb, "only_b": ~ea & eb,
        "both_skip": ~ea & ~eb,
    }
    out = {"models": [name_a, name_b]}
    for g, m in groups.items():
        sub_a = a[m]
        out[g] = {"n": int(m.sum()),
                  f"gross_{name_a}": round(float(sub_a["gross"].sum()), 4),
                  f"gross_{name_b}": round(float(b.loc[m, "gross"].sum()), 4)}
    return out
