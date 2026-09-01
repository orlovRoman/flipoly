"""
MRF T13: Feature diagnostics — dump actual ret/eff/up_ratio values
to understand why no trends are detected.
"""
import sys, os, json, time
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.expanduser("~/flipoly-worktrees/market-regime-filter-implementation"))
from polyflip.crypto.market_regime import build_regime_snapshot, MIN_HISTORY_CANDLES

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

def fetch_candles(symbol, limit=1000, end_time=None):
    import urllib.request, urllib.parse
    params = {"symbol": symbol, "interval": "15m", "limit": limit}
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
        batch = fetch_candles(symbol, 1000, end)
        if not batch: break
        all_c = batch + all_c
        end = batch[0]["open_time"] - 1
        if len(batch) < 1000: break
        time.sleep(0.2)
    return all_c[-total:]

def main():
    print("MRF T13: Feature Diagnostics")
    print("=" * 70)

    for asset in ASSETS:
        try:
            candles = fetch_all(asset)
        except Exception as e:
            print(f"{asset}: error {e}")
            continue

        if len(candles) < MIN_HISTORY_CANDLES + 20:
            continue

        rets, effs, up_ratios, vol_ratios = [], [], [], []
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

            f = snap.assets[asset]
            rets.append(f.ret_24h)
            effs.append(f.efficiency_24h)
            up_ratios.append(f.up_ratio_24h)
            vol_ratios.append(f.vol_ratio)

        print(f"\n{asset}: {len(rets)} evaluations")
        print(f"  ret_24h:      min={min(rets):.4f} max={max(rets):.4f} "
              f"mean={np.mean(rets):.4f} std={np.std(rets):.4f}")
        print(f"  efficiency:   min={min(effs):.4f} max={max(effs):.4f} "
              f"mean={np.mean(effs):.4f} std={np.std(effs):.4f}")
        print(f"  up_ratio:     min={min(up_ratios):.4f} max={max(up_ratios):.4f} "
              f"mean={np.mean(up_ratios):.4f} std={np.std(up_ratios):.4f}")
        print(f"  vol_ratio:    min={min(vol_ratios):.4f} max={max(vol_ratios):.4f} "
              f"mean={np.mean(vol_ratios):.4f} std={np.std(vol_ratios):.4f}")

        # How often would trends trigger with default thresholds?
        trend_up = sum(1 for r, e, u in zip(rets, effs, up_ratios)
                       if r > 0.02 and e > 0.4 and u > 0.65)
        trend_down = sum(1 for r, e, u in zip(rets, effs, up_ratios)
                         if r < -0.02 and e > 0.4 and u < 0.35)
        print(f"  TREND_UP (default):   {trend_up} ({trend_up/len(rets)*100:.1f}%)")
        print(f"  TREND_DOWN (default): {trend_down} ({trend_down/len(rets)*100:.1f}%)")

        # Check which condition fails most
        fail_ret = sum(1 for r in rets if abs(r) <= 0.02)
        fail_eff = sum(1 for r, e in zip(rets, effs) if abs(r) > 0.02 and e <= 0.4)
        fail_breadth = sum(1 for r, e, u in zip(rets, effs, up_ratios)
                          if abs(r) > 0.02 and e > 0.4
                          and not (u > 0.65 or u < 0.35))
        print(f"  Fail reason (when |ret|>0.02 & eff>0.4):")
        print(f"    ret <= 0.02: {fail_ret} ({fail_ret/len(rets)*100:.1f}%)")
        print(f"    eff <= 0.4:  {fail_eff} ({fail_eff/len(rets)*100:.1f}%)")
        print(f"    breadth:     {fail_breadth} ({fail_breadth/len(rets)*100:.1f}%)")

if __name__ == "__main__":
    main()
