"""
MRF T13: Threshold optimization via grid search.
Tests different RegimeConfig parameters to find thresholds that produce
meaningful regime distribution across 45 days of 15m crypto data.
"""
import sys
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import itertools

import numpy as np

sys.path.insert(0, os.path.expanduser("~/flipoly-worktrees/market-regime-filter-implementation"))

from polyflip.crypto.market_regime import build_regime_snapshot, MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_classifier import (
    classify_asset_regime, Regime, RegimeConfig,
)

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# ── Fetch ──────────────────────────────────────────────────────────────────
def fetch_candles(symbol, interval="15m", limit=1000, end_time=None):
    import urllib.request, urllib.parse
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    url = f"https://api.binance.com/api/v3/klines?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "MRF-T13/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return [
        {"open_time": k[0], "open": float(k[1]), "high": float(k[2]),
         "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
        for k in data
    ]

def fetch_all(symbol, total=4320):
    all_c, end = [], None
    while len(all_c) < total:
        batch = fetch_candles(symbol, "15m", 1000, end)
        if not batch: break
        all_c = batch + all_c
        end = batch[0]["open_time"] - 1
        if len(batch) < 1000: break
        time.sleep(0.2)
    return all_c[-total:]

# ── Evaluate config ────────────────────────────────────────────────────────
def evaluate_config(candles_by_asset, config):
    """Run classification with given config, return regime distribution."""
    counts = {r.value: 0 for r in Regime}
    total = 0
    
    for asset, candles in candles_by_asset.items():
        if len(candles) < MIN_HISTORY_CANDLES + 20:
            continue
        window = MIN_HISTORY_CANDLES + 20
        step = 4
        
        for i in range(window, len(candles), step):
            chunk = candles[i - window:i + 1]
            closes = np.array([c["close"] for c in chunk], dtype=np.float64)
            highs = np.array([c["high"] for c in chunk], dtype=np.float64)
            lows = np.array([c["low"] for c in chunk], dtype=np.float64)
            opens = np.array([c["open"] for c in chunk], dtype=np.float64)
            as_of = datetime.fromtimestamp(chunk[-1]["open_time"] / 1000, tz=timezone.utc)
            
            snap = build_regime_snapshot(
                {asset: {"closes": closes, "highs": highs, "lows": lows,
                         "opens": opens, "count": len(closes)}},
                as_of=as_of,
            )
            if not snap.basket.history_ready:
                continue
            
            cl = classify_asset_regime(snap.assets[asset], config=config)
            counts[cl.regime.value] += 1
            total += 1
    
    return counts, total

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("MRF T13: Threshold Optimization")
    print("=" * 60)
    
    # Fetch data once
    print("Fetching candles...")
    candles_by_asset = {}
    for asset in ASSETS:
        try:
            candles_by_asset[asset] = fetch_all(asset)
            print(f"  {asset}: {len(candles_by_asset[asset])} candles")
        except Exception as e:
            print(f"  {asset}: error {e}")
    
    # Default config
    default_cfg = RegimeConfig()
    counts, total = evaluate_config(candles_by_asset, default_cfg)
    print(f"\nDEFAULT config: {total} evaluations")
    for r, c in sorted(counts.items(), key=lambda x: -x[1]):
        if c > 0:
            print(f"  {r}: {c} ({c/total*100:.1f}%)")
    
    # Grid search over key thresholds
    print("\n--- Grid Search ---")
    grid = {
        "trend_ret_threshold": [0.005, 0.01, 0.015, 0.02, 0.03],
        "sideways_ret_max": [0.003, 0.005, 0.008, 0.01],
        "trend_efficiency_min": [0.2, 0.3, 0.4, 0.5],
    }
    
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    
    best_score = -1
    best_params = None
    best_counts = None
    results = []
    
    for combo in combos:
        params = dict(zip(keys, combo))
        cfg = RegimeConfig(**params)
        counts, total = evaluate_config(candles_by_asset, cfg)
        
        if total == 0:
            continue
        
        # Score: want a mix of regimes, not all MIXED
        trend_pct = (counts.get("TREND_UP", 0) + counts.get("TREND_DOWN", 0)) / total
        side_pct = counts.get("SIDEWAYS", 0) / total
        mixed_pct = counts.get("MIXED", 0) / total
        chop_pct = counts.get("HIGH_VOL_CHOP", 0) / total
        
        # Want: some trends (10-30%), some sideways, not too much MIXED
        score = 0
        if 0.05 < trend_pct < 0.40:
            score += 30  # reward having some trends
        elif trend_pct > 0:
            score += 10
        if side_pct > 0.1:
            score += 10
        if mixed_pct < 0.5:
            score += 20  # penalize too much MIXED
        if chop_pct < 0.2:
            score += 10
        
        results.append({"params": params, "counts": counts, "total": total,
                        "trend_pct": trend_pct, "mixed_pct": mixed_pct, "score": score})
        
        if score > best_score:
            best_score = score
            best_params = params
            best_counts = counts
    
    # Top 5 results
    results.sort(key=lambda x: -x["score"])
    print(f"\nTop 5 parameter combinations (of {len(combos)}):")
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1} score={r['score']}, trend={r['trend_pct']*100:.1f}%, mixed={r['mixed_pct']*100:.1f}%")
        print(f"    params: {r['params']}")
        for regime, cnt in sorted(r["counts"].items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"    {regime}: {cnt} ({cnt/r['total']*100:.1f}%)")
    
    if best_params:
        print(f"\n{'='*60}")
        print(f"BEST CONFIG:")
        print(f"  trend_ret_threshold: {best_params['trend_ret_threshold']}")
        print(f"  sideways_ret_max: {best_params['sideways_ret_max']}")
        print(f"  trend_efficiency_min: {best_params['trend_efficiency_min']}")
        print(f"  Score: {best_score}")
        for regime, cnt in sorted(best_counts.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"  {regime}: {cnt} ({cnt/total*100:.1f}%)")
    
    # Save
    output = Path("/tmp/mrf_t13_results.json")
    with open(output, "w") as f:
        json.dump({"best_params": best_params, "best_score": best_score,
                    "top5": [{"params": r["params"], "counts": r["counts"],
                              "score": r["score"], "trend_pct": r["trend_pct"]}
                             for r in results[:5]],
                    "default_counts": counts, "default_total": total,
                    "ts": datetime.now(timezone.utc).isoformat()},
                   f, indent=2, default=str)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
