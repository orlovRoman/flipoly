"""PTM step 22 (CT part): token-regime state per ledger entry + C0 vs C0+CT.

CT_PASS = classify_local_regime(YES-mid history, 15min lookback) == REVERSION.
History source: market_snapshots (SELECT only). Writes ct_states.csv sidecar
keyed by ledger row id + comparison_with_ct.csv. Ledger CSV is immutable.
"""
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

RUN_ID = sys.argv[1]
BASE = os.path.join("artifacts", "research", "price_time_map", RUN_ID)


async def main():
    import asyncpg
    sys.path.insert(0, os.path.abspath("."))
    from polyflip.research.regime_features import classify_local_regime
    rows = []
    with open(os.path.join(BASE, "opportunity_ledger.csv"), newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r.get("selection_status") == "OK" and r.get("side") == "YES":
                r["_lid"] = i
                rows.append(r)
    print("ct candidates:", len(rows))
    bym = {}
    for r in rows:
        bym.setdefault(r["market_id"], []).append(r)
    mids = sorted(bym)
    con = await asyncpg.connect(user="polyflip", password=os.environ.get("PTM_PGPASSWORD", ""), database="polyflip", host="localhost")
    states = {}
    try:
        B = 400
        for i in range(0, len(mids), B):
            chunk = mids[i:i + B]
            db = await con.fetch(
                "SELECT market_id, recorded_at, mid_price FROM market_snapshots"
                " WHERE market_id = ANY($1::text[]) ORDER BY market_id, recorded_at", chunk)
            hist = {}
            for x in db:
                hist.setdefault(x["market_id"], []).append((x["recorded_at"], x["mid_price"]))
            for mid in chunk:
                series = sorted(hist.get(mid, []))
                ts = [t for t, _ in series]
                px = [p for _, p in series]
                for r in bym.get(mid, []):
                    dec = datetime.fromisoformat(r["decision_at"])
                    window = [p for t, p in zip(ts, px) if dec - timedelta(minutes=15) <= t <= dec and p is not None]
                    if len(window) >= 3:
                        try:
                            st = classify_local_regime(window, min_observations=3)
                            states[r["_lid"]] = st.get("state", "UNCERTAIN")
                        except Exception:
                            states[r["_lid"]] = "ERROR"
                    else:
                        states[r["_lid"]] = "INSUFFICIENT_HISTORY"
            print("ct chunk %d/%d" % (i // B + 1, (len(mids) + B - 1) // B), flush=True)
    finally:
        await con.close()
    with open(os.path.join(BASE, "ct_states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lid", "market_id", "recorded_at", "ct_state"])
        w.writeheader()
        for r in rows:
            w.writerow({"lid": r["_lid"], "market_id": r["market_id"],
                        "recorded_at": r["recorded_at"], "ct_state": states.get(r["_lid"], "MISSING")})
    # comparison C0 vs C0+CT on YES_OUTSIDER rows
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r["entry_variant"] == "YES_OUTSIDER" and r["_lid"] in states:
            groups[(r["asset"], r.get("entry_policy"), r["price_bin"])].append(r)
    comp = []
    for key in sorted(groups, key=str):
        rs = groups[key]
        base = sum(float(r["net_pnl"]) for r in rs)
        kept = [r for r in rs if states[r["_lid"]] == "REVERSION"]
        kpnl = sum(float(r["net_pnl"]) for r in kept)
        prevented = sum(-float(r["net_pnl"]) for r in rs if states[r["_lid"]] != "REVERSION" and float(r["net_pnl"]) < 0)
        missed = sum(float(r["net_pnl"]) for r in rs if states[r["_lid"]] != "REVERSION" and float(r["net_pnl"]) > 0)
        comp.append({"asset": key[0], "entry_policy": key[1], "price_bin": key[2],
                     "n_c0": len(rs), "net_c0": round(base, 4),
                     "n_ct": len(kept), "net_ct": round(kpnl, 4),
                     "prevented_losses": round(prevented, 4), "missed_wins": round(missed, 4),
                     "net_diff": round(kpnl - base, 4)})
    with open(os.path.join(BASE, "comparison_with_ct.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)
    import collections as C
    print("ct states:", dict(C.Counter(states.values())))
    print("OUTDIR=" + BASE)


asyncio.run(main())
