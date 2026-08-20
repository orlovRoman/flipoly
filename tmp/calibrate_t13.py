"""
MRF T13: Final threshold calibration + comparison backtest.
Tests relaxed thresholds suitable for 15-minute crypto data.
"""
import sys, os, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.expanduser("~/flipoly-worktrees/market-regime-filter-implementation"))
from polyflip.crypto.market_regime import build_regime_snapshot, MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_classifier import (
    classify_asset_regime, Regime, RegimeConfig,
)

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

def fetch_candles(symbol, limit=1000, end_time=None):
    import urllib.request, urllib.parse
    params = {"symbol": symbol, "interval": "15m", "limit": limit}
    if end_time:
        params["endTime"] = end_time
    url = f"https://api.binance.com/api/v3/klines?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "MRF-T13/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return [
            {"open_time": k[0], "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in json.loads(resp.read())
        ]

def fetch_all(symbol, total=4320):
    all_c, end = [], None
    while len(all_c) < total:
        batch = fetch_candles(symbol, 1000, end)
        if not batch: break
        all_c = batch + all_c
        end = batch[0]["open_time"] - 1
        if len(batch) < 1000: break
        time.sleep(0.2)
    return all_c[-total:]

# ── Configs ──────────────────────────────────────────────────────────────
CONFIGS = {
    "default": RegimeConfig(),
    "relaxed_v1": RegimeConfig(
        trend_ret_threshold=0.01,
        sideways_ret_max=0.005,
        trend_efficiency_min=0.15,
        sideways_efficiency_max=0.12,
        high_vol_efficiency_max=0.10,
        breadth_strong_threshold=0.58,
        breadth_weak_threshold=0.42,
    ),
    "relaxed_v2": RegimeConfig(
        trend_ret_threshold=0.015,
        sideways_ret_max=0.005,
        trend_efficiency_min=0.18,
        sideways_efficiency_max=0.12,
        high_vol_efficiency_max=0.10,
        breadth_strong_threshold=0.57,
        breadth_weak_threshold=0.43,
    ),
    "conservative": RegimeConfig(
        trend_ret_threshold=0.025,
        sideways_ret_max=0.008,
        trend_efficiency_min=0.20,
        sideways_efficiency_max=0.15,
        high_vol_efficiency_max=0.12,
        breadth_strong_threshold=0.56,
        breadth_weak_threshold=0.44,
    ),
}

# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("MRF T13: Threshold Calibration + Comparison")
    print("=" * 70)

    print("Fetching candles...")
    candles_by_asset = {}
    for asset in ASSETS:
        try:
            candles_by_asset[asset] = fetch_all(asset)
            print(f"  {asset}: {len(candles_by_asset[asset])}")
        except Exception as e:
            print(f"  {asset}: error {e}")

    results = {}
    for name, cfg in CONFIGS.items():
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

                cl = classify_asset_regime(snap.assets[asset], config=cfg)
                counts[cl.regime.value] += 1
                total += 1

        results[name] = {"counts": counts, "total": total}
        print(f"\n{name}: {total} evaluations")
        for regime, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"  {regime}: {cnt} ({cnt/total*100:.1f}%)")

    # ── Backtest comparison (using default PolicyConfig for all) ──────────
    print("\n" + "=" * 70)
    print("BACKTEST: BASELINE vs ACTIVE (per RegimeConfig)")
    print("=" * 70)

    for name, regime_cfg in CONFIGS.items():
        baseline_trades = 0
        baseline_pnl = 0.0
        active_trades = 0
        active_pnl = 0.0
        blocked = 0

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

                # Classify with custom config
                cl = classify_asset_regime(snap.assets[asset], config=regime_cfg)

                mock_price = 0.3 + (hash(f"{asset}_{i}") % 100) / 100.0 * 0.4
                is_fav = mock_price > 0.5
                action = "BUY_YES" if is_fav else "BUY_NO"

                # Deterministic win (same for all configs)
                won = hash(f"win_{asset}_{i}") % 100 < 55
                trade_pnl = (0.5 if won else -1.0)

                # Baseline
                baseline_trades += 1
                baseline_pnl += trade_pnl

                # Active: block TREND_UP when outsider, block HIGH_VOL_CHOP
                is_outsider = not is_fav
                blocked_regime = False
                if cl.regime == Regime.TREND_UP and is_outsider:
                    blocked_regime = True
                elif cl.regime == Regime.HIGH_VOL_CHOP:
                    blocked_regime = True
                elif cl.regime == Regime.MIXED:
                    pass  # allow with reduced stake
                elif cl.regime == Regime.SIDEWAYS and is_outsider:
                    pass  # allow
                elif cl.regime == Regime.TREND_DOWN and is_fav:
                    blocked_regime = True

                if not blocked_regime:
                    active_trades += 1
                    active_pnl += trade_pnl
                else:
                    blocked += 1

        print(f"\n{name}:")
        print(f"  BASELINE: {baseline_trades} trades, PnL={baseline_pnl:+.2f}")
        print(f"  ACTIVE:   {active_trades} trades (blocked {blocked}), PnL={active_pnl:+.2f}")
        delta = active_pnl - baseline_pnl
        print(f"  DELTA:    {delta:+.2f}")

    # Save
    output = Path("/tmp/mrf_t13_calibrated.json")
    with open(output, "w") as f:
        json.dump({"configs": {k: {"counts": v["counts"], "total": v["total"]}
                               for k, v in results.items()},
                   "ts": datetime.now(timezone.utc).isoformat()},
                  f, indent=2, default=str)
    print(f"\nSaved: {output}")

if __name__ == "__main__":
    main()
