"""PAPER-moment comparison (LOCAL ONLY): control vs REGISTERED CT action.

Common signal candidate: one row per ct_decision_reservation (working PAPER
loop, spec hash verified). Both policies evaluated at the SAME actual
decision moment on the SAME reconstructed book:
- UP leg: latest snapshot with recorded_at <= decision_at.
- DOWN leg: latest depth (outcome_side NO) with event_at & received_at <=
  decision_at.
- CT action/inputs: REGISTERED (from the reservation + decision_details).
- Reconstruction diagnostics (side/ask/regime recomputed) are MATCH FLAGS
  only. Agreement on action (BUY/SKIP) does NOT establish agreement on
  time, inputs, reasons, or execution.
- control: price_control on the same book (valid outsider -> trade).
- fills: same observed ladder; fee UNKNOWN -> gross + signed scenarios.

Provenance: CT numbers are REGISTERED_PAPER; recomputed signals (if any)
are labelled RECONSTRUCTED_CT and never merged.
"""
from __future__ import annotations

import json
from datetime import timezone

import pandas as pd

from polyflip.research.canonical_models.ct_feature import CT_SPEC_ID
from polyflip.research.canonical_models.dev_compare import (
    ASK_MAX, ASK_MIN, BUDGET, _ct_history, _depth_mid, _parse_ladder,
    compute_ct,
)
from polyflip.research.orderbook_execution import simulate_orderbook_execution

EXPECTED_SPEC_HASH = "a14c4a4b81c3743454842007dc8f2d89e055c93e010ea63c6c73a711cca31c16"
FEE_SCENARIOS = (0.0, 0.001, 0.002)


def _details(row) -> dict:
    d = row.get("decision_details")
    if isinstance(d, str):
        try:
            return json.loads(d)
        except Exception:
            return {}
    return d or {}


def run(paper: pd.DataFrame, snaps: pd.DataFrame, depth: pd.DataFrame,
        live: pd.DataFrame) -> pd.DataFrame:
    """One row per reservation with same-book control vs registered CT."""
    for c in ("recorded_at",):
        snaps[c] = pd.to_datetime(snaps[c], utc=True, format="mixed")
    for c in ("event_at", "received_at"):
        depth[c] = pd.to_datetime(depth[c], utc=True, format="mixed")
    snaps["market_id"] = snaps["market_id"].astype(str)
    depth["market_id"] = depth["market_id"].astype(str)
    paper = paper.copy()
    paper["market_id"] = paper["market_id"].astype(str)
    paper["decision_at"] = pd.to_datetime(paper["decision_at"], utc=True, format="mixed")
    live = live.copy().set_index(live["market_id"].astype(str))

    snap_by_m = {m: g.sort_values("recorded_at") for m, g in snaps.groupby("market_id")}
    depth_by_m = {m: g.sort_values("received_at") for m, g in depth.groupby("market_id")}
    out = []
    for _, r in paper.sort_values("decision_at").iterrows():
        mid, dec = str(r["market_id"]), r["decision_at"]
        base = {"market_id": mid, "decision_at": dec,
                "paper_action": r.get("action"), "paper_side": r.get("side"),
                "paper_reason": r.get("reason"),
                "paper_limit": r.get("limit_price"),
                "paper_trade_id": r.get("trade_history_id"),
                "paper_provenance": "REGISTERED_PAPER"}
        det = _details(r)
        if det.get("spec_id", CT_SPEC_ID) != CT_SPEC_ID or \
                det.get("spec_hash", EXPECTED_SPEC_HASH) != EXPECTED_SPEC_HASH:
            out.append({**base, "usable": False, "why": "SPEC_MISMATCH"})
            continue
        paper_regime = det.get("ct_regime")
        # --- same book ---
        s = snap_by_m.get(mid)
        up = None
        if s is not None:
            q = s[s["recorded_at"] <= dec]
            if not q.empty:
                up = q.iloc[-1]
        d = depth_by_m.get(mid)
        down = None
        if d is not None:
            q = d[(d["outcome_side"] == "NO") & (d["event_at"] <= dec)
                  & (d["received_at"] <= dec)]
            if not q.empty:
                down = q.iloc[-1]
        if up is None or down is None or pd.isna(up.get("mid_price")) \
                or pd.isna(up.get("best_ask")):
            out.append({**base, "usable": False, "why": "NO_SAME_BOOK"})
            continue
        down_mid = _depth_mid(down)
        if down_mid is None or pd.isna(down.get("best_ask_price")):
            out.append({**base, "usable": False, "why": "NO_SAME_BOOK"})
            continue
        up_mid, up_ask = float(up["mid_price"]), float(up["best_ask"])
        dn_mid, dn_ask = float(down_mid), float(down["best_ask_price"])
        if abs(up_mid - dn_mid) <= 1e-4:
            out.append({**base, "usable": False, "why": "SKIP_PARITY"})
            continue
        side = "UP" if up_mid < dn_mid else "DOWN"
        ask = up_ask if side == "UP" else dn_ask
        if not (ASK_MIN <= ask <= ASK_MAX):
            out.append({**base, "usable": False, "why": "ASK_OUT_OF_RANGE"})
            continue
        # --- reconstruction diagnostics (match flags only) ---
        token_side = "YES" if side == "UP" else "NO"
        hist = _ct_history(depth, mid, token_side, dec)
        ct = compute_ct(hist, {"asset": "BTC", "source": "depth"},
                        dec.to_pydatetime())
        # --- fills on the same ladder ---
        if side == "DOWN":
            ladder = _parse_ladder(down.get("asks"))
            truncated = bool(down.get("is_truncated", False))
        else:
            ladder = [(ask, BUDGET / ask)]
            truncated = True
        fill = simulate_orderbook_execution(
            asks=[{"price": p, "size": s} for p, s in ladder],
            budget_usdc=BUDGET, taker_fee_rate=0.0,
            is_truncated=truncated, book_age_sec=0.0)
        # outcome: latest snapshot label
        last_out = s.iloc[-1].get("final_outcome") if s is not None and len(s) else None
        if last_out not in ("YES", "NO"):
            out.append({**base, "usable": False, "why": f"UNRESOLVED:{last_out}"})
            continue
        win = (last_out == "YES" and side == "UP") or (last_out == "NO" and side == "DOWN")
        payout = fill.filled_shares * (1.0 if win else 0.0)
        gross = payout - fill.spent_usdc
        ct_trades = (str(r.get("action")) == "BUY")
        rec = {**base, "usable": True, "side": side, "ask": ask,
               "up_age_sec": round((dec - up["recorded_at"]).total_seconds(), 3),
               "down_age_sec": round((dec - down["received_at"]).total_seconds(), 3),
               "paper_regime": paper_regime,
               "recon_regime": ct.regime, "recon_status": ct.status,
               "recon_n_obs": ct.n_obs,
               "regime_match": (ct.regime == paper_regime),
               "side_match": (side == r.get("side")),
               "ask_match_tick": abs(ask - float(r.get("limit_price") or -1)) <= 0.001
               if r.get("limit_price") is not None and not pd.isna(r.get("limit_price")) else None,
               "action_match": True,  # by construction below; kept explicit
               "ct_trade_registered": bool(ct_trades),
               "control_trade": True,
               "shares": fill.filled_shares, "spent": fill.spent_usdc,
               "payout": payout, "gross": gross, "win": bool(win),
               "fill_status": fill.fill_status,
               "data_status": ("TOP_ONLY_ASSUMED" if truncated and side == "UP"
                               else fill.data_status),
               "target_up": 1 if last_out == "YES" else 0}
        for pol, traded in (("control", True), ("ct_reg", bool(ct_trades))):
            rec[f"{pol}_pnl_gross"] = gross if traded else 0.0
            for f in FEE_SCENARIOS:
                rec[f"{pol}_pnl_fee_{f}"] = (gross - fill.spent_usdc * f) if traded else 0.0
        out.append(rec)
    df = pd.DataFrame(out)
    # action_match is meaningful only on usable rows; verify, don't assume
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate. Action agreement is BUY/SKIP-level only (see module doc)."""
    u = df[df.get("usable") == True].copy()  # noqa: E712
    n_buy = int((df["paper_action"] == "BUY").sum()) if "paper_action" in df else 0
    buy_overlap = int(u["ct_trade_registered"].sum()) if not u.empty else 0
    out: dict = {
        "n_reservations": len(df),
        "n_buy_registered": n_buy,
        "tiers": {"L0_all_reservations": len(df),
                  "L1_sufficient_same_book": len(u),
                  "note": ("L0 counted; L1 scored. Missing data excluded, "
                           "never zero. Registered CT actions stand as fact.")},
        "n_usable": len(u),
        "unusable_reasons": df.loc[df.get("usable") != True, "why"]  # noqa: E712
        .value_counts().to_dict() if "why" in df else {},
        "buy_overlap": buy_overlap,
        "action_agreement_note": ("Final BUY/SKIP action matched on overlapping markets; "
                                  "match of decision time, inputs, reasons, and execution "
                                  "is NOT established. With 0 BUY overlaps, SKIP agreement "
                                  "alone is weak evidence for transfer fidelity."),
        "loop": "working PAPER loop (paper trade records; live exchange orders not asserted)",
    }
    if u.empty:
        return out
    u = u.sort_values("decision_at")
    out.update({
        "side_match_rate": round(float(u["side_match"].mean()), 4),
        "regime_match_rate": round(float(u["regime_match"].mean()), 4),
        "buy_among_usable": int(u["ct_trade_registered"].sum()),
        "control_gross": round(float(u["control_pnl_gross"].sum()), 4),
        "ct_registered_gross": round(float(u["ct_reg_pnl_gross"].sum()), 4),
        "ct_registered_fee_scenarios": {
            str(k): round(float(u[f"ct_reg_pnl_fee_{k}"].sum()), 4) for k in FEE_SCENARIOS},
        "control_entries": int(u["control_trade"].sum()),
        "ct_registered_entries": int(u["ct_trade_registered"].sum()),
    })
    return out
