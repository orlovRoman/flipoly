"""PTM v2 ledger build from frozen inputs (items 3,4,5,6,10,11,18).

Reads: _freeze/<name>/ + observations + expirations. No DB access.
Output: price_time_map/<run_id>/ v2 schema.
"""
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptm_lib2 import (
    ENTRY_TARGETS_MAIN,
    SCENARIO_FEE_RATE,
    bin_of,
    classify_role,
    config_hash,
    opportunity_id,
    pnl_row,
    region_of,
    select_entry,
)

FREEZE = sys.argv[1]
RUN_ID = sys.argv[2]
UNIVERSE = sys.argv[3]  # funnel | independent
OUTDIR = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
STAKE = 1.0
ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE")
COLS = ["opportunity_id", "market_id", "asset", "entry_policy", "entry_rule",
        "target_at", "decision_at", "entry_delay_sec", "time_left_target", "time_left_actual",
        "time_left_from_expiry", "timebase_status",
        "entry_variant", "side", "role", "role_basis",
        "mid_price", "best_bid", "best_ask", "spread", "price_bin", "bin_status", "region",
        "final_outcome", "target", "stake_usdc", "shares",
        "gross_pnl", "fee_status", "fee", "net_pnl", "scenario_fee", "scenario_net_pnl",
        "quote_source", "selection_status", "skip_reason",
        "recorded_at", "calendar_date", "hour_utc", "weekday"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True).stdout.strip()


def research_file_hashes():
    out = {}
    for name in ("ptm_lib2.py", "ptm_build2.py", "ptm_analyze2.py", "ptm_freeze.py"):
        p = os.path.join("scripts", "research", name)
        if os.path.exists(p):
            out[name] = sha256_file(p)
    return out


def load_freeze():
    base = os.path.join("artifacts", "research", "price_time_map", "_freeze", FREEZE)
    meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
    snaps = {}
    for f in meta["files"]:
        with gzip.open(os.path.join(base, f["path"]), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                r["recorded_at"] = datetime.fromisoformat(str(r["recorded_at"]))
                if r["recorded_at"].tzinfo is None:
                    r["recorded_at"] = r["recorded_at"].replace(tzinfo=timezone.utc)
                r["time_left_min"] = float(r["time_left_min"]) if r["time_left_min"] else None
                for k in ("mid_price", "best_bid", "best_ask", "spread"):
                    r[k] = float(r[k]) if r[k] not in (None, "") else None
                snaps.setdefault(r["market_id"], []).append(r)
    return meta, snaps


def main():
    t0 = datetime.now(timezone.utc)
    obs_doc = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))
    obs = [o for o in obs_doc["observations"] if o.get("asset") in ASSETS]
    exp = json.load(open("artifacts/research/market_expirations.json", encoding="utf-8"))
    mids = sorted({str(o["market_id"]) for o in obs})
    period = sorted(o["timestamp"] for o in obs if o.get("timestamp"))
    asset_of = {}
    for o in obs:
        asset_of.setdefault(str(o["market_id"]), o.get("asset"))
    job_of = {}
    for o in obs:
        job_of.setdefault(str(o["market_id"]), o)
    config = {
        "stake_usdc": STAKE, "entry_time_grid": list(ENTRY_TARGETS_MAIN),
        "entry_delay_max_sec": 30.0, "price_bins": "plan-14-fixed+v2-out-of-range",
        "fee_model": "gross + scenario_fee", "fee_status": "UNKNOWN",
        "scenario_fee_rate": SCENARIO_FEE_RATE,
        "side_definition": "unified role by mid (basis MID); ask-only fallback marked ASK_ONLY",
        "outcome_definition": "frozen market_snapshots.final_outcome",
        "bootstrap_seed": 20260911, "assets": list(ASSETS),
        "period_utc": [period[0], period[-1]],
        "freeze": FREEZE, "universe": UNIVERSE,
    }
    manifest = {**config, "config_hash": config_hash(config), "run_id": RUN_ID,
                "base_commit": "926fb57", "research_commit": git("rev-parse", "HEAD"),
                "dirty_worktree": bool(git("status", "--short")),
                "research_file_hashes": research_file_hashes(),
                "input_file_hashes": {
                    "observations_30d.json": sha256_file("artifacts/weighted_policy/observations_30d.json"),
                    "market_expirations.json": sha256_file("artifacts/research/market_expirations.json")}}
    meta, snaps_by_m = load_freeze()
    manifest["freeze_meta"] = meta["name"]
    cov = {"markets": len(mids), "with_snapshots": 0, "outcome_ok": 0, "outcome_conflict": 0,
           "outcome_missing": 0, "entries_ok": 0, "entries_missing": 0,
           "timebase_mismatch": 0, "per_asset": {}}
    os.makedirs(OUTDIR, exist_ok=True)
    f = open(os.path.join(OUTDIR, "opportunity_ledger.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    n = 0
    for mid in mids:
        snaps = snaps_by_m.get(mid)
        if not snaps:
            continue
        cov["with_snapshots"] += 1
        outs = {s["final_outcome"] for s in snaps if s["final_outcome"]}
        asset = asset_of.get(mid)
        if not outs:
            cov["outcome_missing"] += 1
            continue
        if len(outs) > 1:
            cov["outcome_conflict"] += 1
            continue
        cov["outcome_ok"] += 1
        final = outs.pop()
        if mid not in exp:
            continue
        expiry = datetime.fromisoformat(exp[mid])
        for x in ENTRY_TARGETS_MAIN:
            target_at = expiry - timedelta(minutes=x)
            base = {"market_id": mid, "asset": asset, "entry_policy": "GRID",
                    "entry_rule": "T-%d" % x, "target_at": target_at.isoformat(),
                    "time_left_target": x, "final_outcome": final}
            row, status, reason, delay = select_entry(snaps, target_at)
            if status != "OK":
                cov["entries_missing"] += 1
                w.writerow({**base, "decision_at": target_at.isoformat(), "entry_delay_sec": "",
                            "selection_status": status, "skip_reason": reason})
                n += 1
                continue
            dt = row["recorded_at"]
            tl_db = row["time_left_min"]
            tl_exp = (expiry - dt).total_seconds() / 60.0
            tb = "OK" if abs((tl_db or 0) - tl_exp) * 60.0 < 120.0 else "TIMEBASE_MISMATCH"
            if tb != "OK":
                cov["timebase_mismatch"] += 1
            cov["entries_ok"] += 1
            mid_px, bid, ask = row["mid_price"], row["best_bid"], row["best_ask"]
            if mid_px is None or ask is None:
                cov["entries_missing"] += 1
                w.writerow({**base, "decision_at": dt.isoformat(), "entry_delay_sec": round(delay, 1),
                            "selection_status": "MISSING_ENTRY_QUOTE", "skip_reason": "NO_REAL_QUOTE"})
                n += 1
                continue
            role, basis = classify_role(mid_px, ask)
            variant = ("YES_OUTSIDER" if role == "OUTSIDER" else "YES_FAVORITE")
            won = (final == "YES")
            binlabel, binstatus = bin_of(ask)
            shares, cash, gross, fee, net, fs = pnl_row(ask, won, STAKE, None)
            _, _, _, sfee, snet, _ = pnl_row(ask, won, STAKE, SCENARIO_FEE_RATE)
            oid = opportunity_id(mid, "GRID", "T-%d" % x, "YES", dt.isoformat())
            w.writerow({
                "opportunity_id": oid, **base, "decision_at": dt.isoformat(),
                "entry_delay_sec": round(delay, 1),
                "time_left_actual": round(tl_db if tl_db is not None else tl_exp, 3),
                "time_left_from_expiry": round(tl_exp, 3), "timebase_status": tb,
                "entry_variant": variant, "side": "YES", "role": role, "role_basis": basis,
                "mid_price": mid_px, "best_bid": bid, "best_ask": ask,
                "spread": row.get("spread"), "price_bin": binlabel or "", "bin_status": binstatus,
                "region": region_of(ask),
                "target": int(won), "stake_usdc": STAKE,
                "shares": round(shares, 6), "gross_pnl": round(gross, 6),
                "fee_status": "UNKNOWN", "fee": "", "net_pnl": "",
                "scenario_fee": round(sfee, 6), "scenario_net_pnl": round(snet, 6),
                "quote_source": "SNAPSHOT_ASK",
                "selection_status": "OK", "skip_reason": "",
                "recorded_at": dt.isoformat(),
                "calendar_date": dt.date().isoformat(), "hour_utc": dt.hour, "weekday": dt.weekday(),
            })
            n += 1
        # FUNNEL stratum (real funnel quotes only)
        o = job_of.get(mid)
        if o is not None and o.get("timestamp"):
            dt = datetime.fromisoformat(str(o["timestamp"]))
            tl = (expiry - dt).total_seconds() / 60.0
            if tl >= 0:
                fvars = []
                if o.get("yes_ask") is not None:
                    fvars.append((o["yes_ask"], "YES", o.get("p_market_yes")))
                if o.get("no_ask") is not None:
                    fvars.append((o["no_ask"], "NO", None))
                if not fvars:
                    cov["entries_missing"] += 1
                    w.writerow({"market_id": mid, "asset": asset, "entry_policy": "FUNNEL",
                                "entry_rule": "FUNNEL_OBSERVED", "target_at": "", "decision_at": dt.isoformat(),
                                "entry_delay_sec": "", "time_left_target": "", "time_left_actual": round(tl, 3),
                                "final_outcome": final, "selection_status": "MISSING_ENTRY_QUOTE",
                                "skip_reason": "NO_REAL_QUOTE_IN_FUNNEL_ROW"})
                    n += 1
                for aq, side, mpx in fvars:
                    role, basis = classify_role(mpx if side == "YES" else None, aq)
                    if side == "NO":
                        role = "OUTSIDER" if aq <= 0.5 else "FAVORITE"
                        basis = "ASK_ONLY"
                    variant = side + "_" + role
                    won = (final == side)
                    shares, cash, gross, fee, net, fs = pnl_row(aq, won, STAKE, None)
                    _, _, _, sfee, snet, _ = pnl_row(aq, won, STAKE, SCENARIO_FEE_RATE)
                    binlabel, binstatus = bin_of(aq)
                    oid = opportunity_id(mid, "FUNNEL", "FUNNEL_OBSERVED", side, dt.isoformat())
                    cov["entries_ok"] += 1
                    w.writerow({
                        "opportunity_id": oid, "market_id": mid, "asset": asset,
                        "entry_policy": "FUNNEL", "entry_rule": "FUNNEL_OBSERVED",
                        "target_at": "", "decision_at": dt.isoformat(), "entry_delay_sec": "",
                        "time_left_target": "", "time_left_actual": round(tl, 3),
                        "time_left_from_expiry": round(tl, 3), "timebase_status": "OK",
                        "entry_variant": variant, "side": side, "role": role, "role_basis": basis,
                        "mid_price": mpx if mpx is not None else "", "best_bid": "", "best_ask": aq,
                        "spread": o.get("spread"), "price_bin": binlabel or "", "bin_status": binstatus,
                        "region": region_of(aq),
                        "target": int(won), "stake_usdc": STAKE,
                        "shares": round(shares, 6), "gross_pnl": round(gross, 6),
                        "fee_status": "UNKNOWN", "fee": "", "net_pnl": "",
                        "scenario_fee": round(sfee, 6), "scenario_net_pnl": round(snet, 6),
                        "quote_source": "FUNNEL_OBSERVED_" + side + "_ASK",
                        "selection_status": "OK", "skip_reason": "",
                        "recorded_at": dt.isoformat(),
                        "calendar_date": dt.date().isoformat(), "hour_utc": dt.hour, "weekday": dt.weekday(),
                    })
                    n += 1
    f.close()
    json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=2)
    json.dump(cov, open(os.path.join(OUTDIR, "coverage.json"), "w"), indent=2, default=str)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print("run:", RUN_ID, "ledger rows:", n, "secs:", round(dt))
    print("OUTDIR=" + OUTDIR)


main()
