"""Development comparison: price_control vs CT on a COMMON sample (LOCAL ONLY).

Reads local export chunks (snapshots + depth + live registry). No DB, no
server, no network. Chunked: callers pass day-filtered frames; CT history
needs the previous day's depth — load it explicitly (see run_days).

Per market (resolved YES/NO only):
- decision = first snapshot with recorded_at in (T-5m, T-4:45m], UP leg
  mid/ask present; DOWN leg from latest depth (event_at & received_at <=
  decision). Missing leg -> SKIP (reason kept, no 1-YES restore).
- outsider = lower mid; parity -> SKIP_PARITY; top ask must sit in
  [0.01, 0.40].
- CT (BTC scope only; else ASSET_OUT_OF_SCOPE): chosen-token depth mids in
  [decision-900s, decision], >=3 obs, via fixed wrapper. Provenance of every
  historical signal: RECONSTRUCTED_CT. Registered PAPER decisions are loaded
  separately for reconciliation and NEVER merged into these numbers.
- fill: observed ask ladder at decision (budget $1 purchase cost) via the
  vendored orderbook engine. Fee UNKNOWN -> gross + signed scenarios
  (0 / 0.001 / 0.002 of purchase cost + crypto-v2 formula, period
  unconfirmed). Outcome YES = UP leg wins.

Common sample: every market with valid decision + valid outsider ask +
known outcome is scored by BOTH policies (trade or SKIP with reason).
"""
from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd

from polyflip.research.canonical_models.ct_feature import (
    CT_SPEC_ID, compute_ct, ct_allows_entry,
)
from polyflip.research.canonical_models.policies import (
    Opportunity, decide, pick_outsider_side,
)
from polyflip.research.orderbook_execution import simulate_orderbook_execution

ASK_MIN, ASK_MAX = 0.01, 0.40
BUDGET = 1.0
CT_HISTORY_SEC = 900.0
PROVENANCE = "RECONSTRUCTED_CT"
EV_THRESHOLD = 0.02  # gross gate, pre-registered (protocol)


def _parse_ladder(cell) -> list[tuple[float, float]]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, str):
        cell = json.loads(cell)
    out = []
    for lvl in cell:
        p, s = float(lvl["price"]), float(lvl["size"])
        if p > 0 and s > 0:
            out.append((p, s))
    return sorted(out)


def _depth_mid(row) -> float | None:
    b, a = row.get("best_bid_price"), row.get("best_ask_price")
    if b is None or a is None or pd.isna(b) or pd.isna(a):
        return None
    b, a = float(b), float(a)
    if not (0.0 < b < 1.0 and 0.0 < a < 1.0) or b > a:
        return None
    return (a + b) / 2.0


def build_decisions(snaps: pd.DataFrame, depth: pd.DataFrame,
                    live: pd.DataFrame) -> pd.DataFrame:
    """One row per market with decision point + both legs (or skip reason)."""
    snaps = snaps.copy()
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"], utc=True, format="mixed")
    depth = depth.copy()
    for c in ("event_at", "received_at"):
        depth[c] = pd.to_datetime(depth[c], utc=True, format="mixed")
    live = live.copy().set_index(live["market_id"].astype(str))
    snaps["market_id"] = snaps["market_id"].astype(str)
    depth["market_id"] = depth["market_id"].astype(str)

    depth_by_market: dict[str, pd.DataFrame] = {
        m: g.sort_values("received_at") for m, g in depth.groupby("market_id")
    }
    out = []
    for mid, g in snaps.groupby("market_id"):
        if mid not in live.index:
            out.append({"market_id": mid, "status": "SKIP", "reason": "NO_REGISTRY"})
            continue
        lm = live.loc[mid]
        end = pd.to_datetime(lm["end_time_est"], utc=True)
        if pd.isna(end):
            out.append({"market_id": mid, "status": "SKIP", "reason": "NO_END"})
            continue
        target = end - timedelta(seconds=300)
        cand = g[(g["recorded_at"] > target)
                 & (g["recorded_at"] <= target + timedelta(seconds=15))]
        cand = cand.dropna(subset=["mid_price", "best_ask"])
        if cand.empty:
            out.append({"market_id": mid, "asset": lm["asset"], "status": "SKIP",
                        "reason": "NO_DECISION_SNAPSHOT"})
            continue
        dec = cand.sort_values("recorded_at").iloc[0]
        decision_at = dec["recorded_at"]
        outcome = dec["final_outcome"]
        if outcome not in ("YES", "NO"):
            out.append({"market_id": mid, "status": "SKIP", "reason": f"UNRESOLVED:{outcome}"})
            continue
        up_mid, up_ask = float(dec["mid_price"]), float(dec["best_ask"])
        if not (0.0 < up_mid < 1.0 and 0.0 < up_ask < 1.0):
            out.append({"market_id": mid, "status": "SKIP", "reason": "BAD_UP_QUOTE"})
            continue
        # DOWN leg: latest NO depth snapshot, causal on both timestamps
        d = depth_by_market.get(mid)
        down = None
        if d is not None:
            no = d[(d["outcome_side"] == "NO") & (d["event_at"] <= decision_at)
                   & (d["received_at"] <= decision_at)]
            if not no.empty:
                down = no.iloc[-1]
        if down is None:
            out.append({"market_id": mid, "asset": lm["asset"], "status": "SKIP",
                        "reason": "NO_DOWN_QUOTE", "decision_at": decision_at})
            continue
        down_mid = _depth_mid(down)
        down_ask = down.get("best_ask_price")
        if down_mid is None or down_ask is None or pd.isna(down_ask):
            out.append({"market_id": mid, "asset": lm["asset"], "status": "SKIP",
                        "reason": "BAD_DOWN_QUOTE", "decision_at": decision_at})
            continue
        out.append({
            "market_id": mid, "asset": lm["asset"], "status": "OK",
            "decision_at": decision_at, "end_at": end,
            "up_mid": up_mid, "up_ask": up_ask,
            "down_mid": float(down_mid), "down_ask": float(down_ask),
            "down_depth_row": down.to_dict(),
            "target_up": 1 if outcome == "YES" else 0,
            "up_token_id": lm.get("yes_token_id"), "down_token_id": lm.get("no_token_id"),
        })
    return pd.DataFrame(out)


def _ct_history(depth: pd.DataFrame, mid: str, token_side: str,
                decision_at: pd.Timestamp) -> list[dict]:
    depth = depth.copy()
    for c in ("event_at", "received_at"):
        depth[c] = pd.to_datetime(depth[c], utc=True, format="mixed")
    d = depth[(depth["market_id"].astype(str) == str(mid))
              & (depth["outcome_side"] == token_side)]
    d = d[(d["event_at"] >= decision_at - timedelta(seconds=CT_HISTORY_SEC))
          & (d["event_at"] <= decision_at) & (d["received_at"] <= decision_at)]
    hist = []
    for _, r in d.sort_values("event_at").iterrows():
        m = _depth_mid(r)
        if m is not None:
            hist.append({"event_at": r["event_at"], "price": m,
                         "received_at": r["received_at"]})
    return hist


def evaluate(decisions: pd.DataFrame, depth: pd.DataFrame,
             fee_scenarios=(0.0, 0.001, 0.002)) -> pd.DataFrame:
    """Score BOTH policies on the common sample. Returns per-market rows."""
    rows = []
    for _, r in decisions.iterrows():
        base = {"market_id": r["market_id"], "asset": r.get("asset"),
                "decision_at": r.get("decision_at"), "target_up": r.get("target_up")}
        if r["status"] != "OK":
            rows.append({**base, "in_sample": False, "reason": r.get("reason")})
            continue
        opp = Opportunity(str(r["market_id"]), float(r["up_mid"]), float(r["down_mid"]),
                          float(r["up_ask"]), float(r["down_ask"]),
                          float(r["up_mid"]), None, "UNCERTAIN")
        side, why = pick_outsider_side(opp)
        if side is None:
            rows.append({**base, "in_sample": False, "reason": why})
            continue
        ask = float(r["up_ask"] if side == "UP" else r["down_ask"])
        if not (ASK_MIN <= ask <= ASK_MAX):
            rows.append({**base, "in_sample": False, "reason": "ASK_OUT_OF_RANGE"})
            continue
        token_side = "YES" if side == "UP" else "NO"
        hist = _ct_history(depth, str(r["market_id"]), token_side, r["decision_at"])
        ct = compute_ct(hist, {"asset": str(r.get("asset")), "source": "depth"},
                        r["decision_at"].to_pydatetime())
        # fill on the observed ladder of the decision-time depth row
        ladder = _parse_ladder(r["down_depth_row"].get("asks")) if side == "DOWN" else None
        if ladder is None:  # UP leg: top-of-book ladder from snapshot ask
            ladder = [(ask, BUDGET / ask)]
            ladder_truncated = True
        else:
            ladder_truncated = bool(r["down_depth_row"].get("is_truncated", False))
        fill = simulate_orderbook_execution(
            asks=[{"price": p, "size": s} for p, s in ladder],
            budget_usdc=BUDGET, taker_fee_rate=0.0,
            is_truncated=ladder_truncated,
            book_age_sec=0.0,
        )
        win = (r["target_up"] == 1 and side == "UP") or (r["target_up"] == 0 and side == "DOWN")
        payout = fill.filled_shares * (1.0 if win else 0.0)
        gross = payout - fill.spent_usdc
        scen = {f"fee_{f}": gross - fill.spent_usdc * f for f in fee_scenarios}
        vwap = fill.vwap or ask
        scen["fee_crypto_v2_formula_unconfirmed"] = (
            gross - fill.filled_shares * 0.07 * vwap * (1.0 - vwap))
        rec = {**base, "in_sample": True, "side": side, "ask": ask,
               "vwap": vwap, "shares": fill.filled_shares,
               "spent": fill.spent_usdc, "payout": payout, "gross": gross,
               "win": bool(win), "fill_status": fill.fill_status,
               "data_status": ("TOP_ONLY_ASSUMED" if ladder_truncated and side == "UP"
                               else fill.data_status),
               "ct_regime": ct.regime, "ct_status": ct.status,
               "ct_n_obs": ct.n_obs, "ct_provenance": PROVENANCE,
               "ct_spec": CT_SPEC_ID, "ct_spec_hash": ct.spec_hash,
               **scen}
        # policies (common sample: both scored)
        d_pc = decide("price_control", opp, EV_THRESHOLD)
        opp_ct = Opportunity(opp.market_id, opp.up_mid, opp.down_mid, opp.up_ask,
                             opp.down_ask, opp.p_up_market, opp.p_up_model, ct.regime)
        d_ct = decide("CT", opp_ct, EV_THRESHOLD)
        rec["control_trade"] = bool(d_pc.trade)
        rec["control_reason"] = d_pc.reason
        rec["ct_trade"] = bool(d_ct.trade) and ct_allows_entry(ct)
        rec["ct_reason"] = ("CT gate evaluated" if rec["ct_trade"]
                            else f"CT blocks: {ct.regime}/{ct.status}")
        for pol, traded in (("control", rec["control_trade"]), ("ct", rec["ct_trade"])):
            rec[f"{pol}_pnl_gross"] = rec["gross"] if traded else 0.0
            for f in fee_scenarios:
                rec[f"{pol}_pnl_fee_{f}"] = rec[f"fee_{f}"] if traded else 0.0
            rec[f"{pol}_pnl_formula"] = rec["fee_crypto_v2_formula_unconfirmed"] if traded else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> dict:
    """Aggregate a common-sample comparison (development only).

    MAIN pair: BTC-only control vs BTC control+CT on IDENTICAL opportunity
    ids (self-checked below). Five-asset control stays as a descriptive
    table only (asset mix would confound the CT effect).
    """
    s = scored[scored["in_sample"] == True].copy()  # noqa: E712
    if s.empty:
        return {"n": 0}
    s = s.sort_values("decision_at")
    b = s[s["asset"] == "BTC"].copy()
    assert len(b) and b["control_trade"].notna().all() and b["ct_trade"].notna().all(), \
        "both policies must be scored on every BTC opportunity"
    out_btc: dict = {
        "n": len(b),
        "opportunity_ids": sorted(b["market_id"].astype(str).unique()),
        "control_gross": round(float(b["control_pnl_gross"].sum()), 4),
        "control_entries": int(b["control_trade"].sum()),
        "ct_gross": round(float(b["ct_pnl_gross"].sum()), 4),
        "ct_entries": int(b["ct_trade"].sum()),
        "diff_gross": round(float((b["ct_pnl_gross"] - b["control_pnl_gross"]).sum()), 4),
    }
    out: dict = {"n": len(s), "provenance": PROVENANCE,
                 "btc_pair_main": out_btc,
                 "common_sample_check": {
                     "identical_opportunity_ids_pre_ct": True,
                     "n_btc_opportunities": out_btc["n"],
                     "scope": "BTC only (CT spec scope); other assets descriptive"},
                 "by_asset": s["asset"].value_counts().to_dict(),
                 "by_side": s["side"].value_counts().to_dict(),
                 "control_entries": int(s["control_trade"].sum()),
                 "control_all_assets_descriptive": "5-asset control is descriptive only; "
                                                  "do not compare with BTC-scoped CT",
                 "ct_entries": int(s["ct_trade"].sum()),
                 "ct_regimes": s["ct_regime"].value_counts().to_dict()}
    for pol in ("control", "ct"):
        pnl = s[f"{pol}_pnl_gross"]
        wins = s[pol + "_trade"] & s["win"]
        out[pol] = {
            "gross": round(float(pnl.sum()), 4),
            "per_opp": round(float(pnl.mean()), 5),
            "entries": int(s[pol + "_trade"].sum()),
            "win_rate": round(float(wins.sum() / max(s[pol + "_trade"].sum(), 1)), 4),
            "fee_scenarios": {str(k): round(float(s[f"{pol}_pnl_fee_{k}"].sum()), 4)
                              for k in (0.0, 0.001, 0.002)},
            "formula_unconfirmed": round(float(s[f"{pol}_pnl_formula"].sum()), 4),
            "max_drawdown": round(float((pnl.cumsum().cummax() - pnl.cumsum()).max()), 4),
            "top_win_share": round(float(pnl.max() / max(pnl.sum(), 1e-9)), 4),
        }
    diff = (s["ct_pnl_gross"] - s["control_pnl_gross"]).tolist()
    out["ct_minus_control_gross"] = round(sum(diff), 4)
    # paired day-block bootstrap, days resampled whole (local)
    import random
    s["day"] = pd.to_datetime(s["decision_at"], utc=True).dt.date.astype(str)
    by_day = {d: g for d, g in s.groupby("day")}
    days = sorted(by_day)
    rng = random.Random(42)
    boots = []
    for _ in range(2000):
        tot = 0.0
        for _ in days:
            g = by_day[rng.choice(days)]
            tot += float((g["ct_pnl_gross"] - g["control_pnl_gross"]).sum())
        boots.append(tot)
    boots.sort()
    out["diff_ci95"] = [round(boots[50], 3), round(boots[1950], 3)]
    out["n_days"] = len(days)
    return out
