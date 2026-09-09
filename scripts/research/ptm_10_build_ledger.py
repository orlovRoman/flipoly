"""PTM steps 3,4,6,7,8,9,10,11: manifest, coverage, opportunity ledger.

Inputs (read-only): artifacts/weighted_policy/observations_30d.json,
artifacts/research/market_expirations.json, market_snapshots (SELECT only).
Output: artifacts/research/price_time_map/<run_id>/{manifest.json,opportunity_ledger.csv,coverage.json}
"""
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptm_lib import (
    ENTRY_TARGETS_MAIN,
    SCENARIO_FEE_RATE,
    bin_of,
    config_hash,
    pnl_row,
    region_of,
    select_entry,
)

RUN_ID = datetime.now(timezone.utc).strftime("ptm_%Y%m%d_%H%M%S")
OUTDIR = os.path.join("artifacts", "research", "price_time_map", RUN_ID)
STAKE = 1.0
ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE")
COLS = ["market_id", "asset", "decision_at", "time_left_target", "time_left_actual", "entry_variant",
        "side", "mid_price", "best_bid", "best_ask", "spread", "price_bin", "region", "final_outcome",
        "target", "stake_usdc", "shares", "gross_pnl", "fee_status", "fee", "net_pnl",
        "scenario_fee", "scenario_net_pnl", "selection_status", "skip_reason",
        "recorded_at", "calendar_date", "hour_utc", "weekday"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True).stdout.strip()


def process_chunk(snaps_by_m, chunk, asset_of, json_outcome, exp, w, cov):
    n = 0
    for mid in chunk:
        snaps = snaps_by_m.get(mid)
        if not snaps:
            continue
        cov["with_snapshots"] += 1
        outs = {s["final_outcome"] for s in snaps if s["final_outcome"] is not None}
        asset = asset_of.get(mid)
        if not outs:
            cov["outcome_missing"] += 1
            continue
        if len(outs) > 1:
            cov["outcome_conflict"] += 1
            continue
        cov["outcome_ok"] += 1
        final = outs.pop()
        if json_outcome.get(mid) is not None:
            cov["json_compared"] += 1
            if json_outcome[mid] == final:
                cov["json_agreement"] += 1
        try:
            expiry = datetime.fromisoformat(exp[mid])
        except KeyError:
            continue
        pa = cov["per_asset"].setdefault(asset, {"markets": 0, "min_tl": 1e9, "max_tl": -1e9})
        pa["markets"] += 1
        for s in snaps:
            tl = s["time_left_min"]
            if tl is not None:
                if tl < pa["min_tl"]:
                    pa["min_tl"] = tl
                if tl > pa["max_tl"]:
                    pa["max_tl"] = tl
        for x in ENTRY_TARGETS_MAIN:
            decision_at = expiry - timedelta(minutes=x)
            base = {"market_id": mid, "asset": asset, "decision_at": decision_at.isoformat(),
                    "time_left_target": x, "final_outcome": final}
            row, status, reason = select_entry(snaps, x, decision_at)
            if status != "OK":
                cov["entries_missing"] += 1
                w.writerow({**base, "time_left_actual": "", "selection_status": status, "skip_reason": reason})
                n += 1
                continue
            cov["entries_ok"] += 1
            mid_px, bid, ask = row["mid_price"], row["best_bid"], row["best_ask"]
            tl_act = row["time_left_min"]
            rec_at = row["recorded_at"]
            variants = []
            if mid_px is not None and ask is not None:
                variants.append(("YES_OUTSIDER" if mid_px <= 0.5 else "YES_FAVORITE", ask, mid_px))
            cov["no_quote_variants"] += 2
            for variant, entry_price, mpx in variants:
                won = (final == "YES")
                shares, cash, gross, fee, net, _ = pnl_row(entry_price, won, STAKE, None)
                _, _, _, sfee, snet, _ = pnl_row(entry_price, won, STAKE, SCENARIO_FEE_RATE)
                dt = rec_at if isinstance(rec_at, datetime) else datetime.fromisoformat(str(rec_at))
                w.writerow({
                    **base, "decision_at": dt.isoformat(), "time_left_actual": round(tl_act, 3),
                    "entry_variant": variant, "side": "YES",
                    "mid_price": mid_px, "best_bid": bid, "best_ask": ask,
                    "spread": row.get("spread"), "price_bin": bin_of(entry_price), "region": region_of(entry_price),
                    "target": int(won), "stake_usdc": STAKE,
                    "shares": round(shares, 6), "gross_pnl": round(gross, 6),
                    "fee_status": "UNKNOWN", "fee": 0.0, "net_pnl": round(gross, 6),
                    "scenario_fee": round(sfee, 6), "scenario_net_pnl": round(snet, 6),
                    "selection_status": "OK", "skip_reason": "",
                    "recorded_at": dt.isoformat(),
                    "calendar_date": dt.date().isoformat(), "hour_utc": dt.hour, "weekday": dt.weekday(),
                })
                n += 1
    return n


async def main():
    import asyncpg
    t0 = datetime.now(timezone.utc)
    obs_doc = json.load(open("artifacts/weighted_policy/observations_30d.json", encoding="utf-8"))
    obs = [o for o in obs_doc["observations"] if o.get("asset") in ASSETS]
    exp = json.load(open("artifacts/research/market_expirations.json", encoding="utf-8"))
    mids = sorted({str(o["market_id"]) for o in obs})
    period = sorted(o["timestamp"] for o in obs if o.get("timestamp"))
    asset_of = {}
    for o in obs:
        asset_of.setdefault(str(o["market_id"]), o.get("asset"))
    json_outcome = {str(o["market_id"]): o.get("outcome_yes") for o in obs}
    config = {
        "run_id": RUN_ID, "stake_usdc": STAKE,
        "entry_time_grid": list(ENTRY_TARGETS_MAIN),
        "entry_window_min": 0.5, "price_bins": "plan-14-fixed",
        "fee_model": "gross_or_observed_quote + scenario_fee", "fee_status": "UNKNOWN",
        "scenario_fee_rate": SCENARIO_FEE_RATE,
        "side_definition": "mid<=0.5 OUTSIDER / mid>0.5 FAVORITE at decision; NO_* requires real NO book",
        "outcome_definition": "DB market_snapshots.final_outcome",
        "bootstrap_seed": 20260911, "assets": list(ASSETS),
        "period_utc": [period[0], period[-1]],
    }
    manifest = {**config, "config_hash": config_hash(config),
                "base_commit": git("rev-parse", "HEAD"), "research_commit": git("rev-parse", "HEAD"),
                "dirty_worktree": bool(git("status", "--short")),
                "input_file_hashes": {
                    "observations_30d.json": sha256_file("artifacts/weighted_policy/observations_30d.json"),
                    "market_expirations.json": sha256_file("artifacts/research/market_expirations.json")}}
    cov = {"markets": len(mids), "with_snapshots": 0, "outcome_ok": 0, "outcome_conflict": 0,
           "outcome_missing": 0, "json_agreement": 0, "json_compared": 0,
           "entries_ok": 0, "entries_missing": 0, "no_quote_variants": 0, "per_asset": {}}
    os.makedirs(OUTDIR, exist_ok=True)
    con = await asyncpg.connect(user="polyflip", password=os.environ.get("PTM_PGPASSWORD", ""), database="polyflip", host="localhost")
    n_rows = 0
    try:
        manifest["db_snapshot_rows"] = await con.fetchval("SELECT count(*) FROM market_snapshots")
        manifest["db_snapshot_max_id"] = await con.fetchval("SELECT max(id) FROM market_snapshots")
        B = 1500
        nch = (len(mids) + B - 1) // B
        f = open(os.path.join(OUTDIR, "opportunity_ledger.csv"), "w", newline="")
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for i in range(0, len(mids), B):
            chunk = mids[i:i + B]
            rows = await con.fetch(
                "SELECT market_id, recorded_at, time_left_min, mid_price, best_bid, best_ask, spread, final_outcome"
                " FROM market_snapshots WHERE market_id = ANY($1::text[]) ORDER BY market_id, recorded_at",
                chunk)
            snaps_by_m = {}
            for r in rows:
                snaps_by_m.setdefault(r["market_id"], []).append(dict(r))
            n_rows += process_chunk(snaps_by_m, chunk, asset_of, json_outcome, exp, w, cov)
            print("chunk %d/%d rows=%d" % (i // B + 1, nch, n_rows), flush=True)
        f.close()
    finally:
        await con.close()
    json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=2)
    json.dump(cov, open(os.path.join(OUTDIR, "coverage.json"), "w"), indent=2, default=str)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print("run:", RUN_ID, "ledger rows:", n_rows, "secs:", round(dt))
    print("OUTDIR=" + OUTDIR)


asyncio.run(main())
