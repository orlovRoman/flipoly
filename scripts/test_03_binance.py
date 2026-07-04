from polyflip.crypto.binance_client import fetch_klines

candles = fetch_klines("BTCUSDT", "15m", limit=10)
assert len(candles) == 10, f"Ожидали 10 свечей, получили {len(candles)}"
c = candles[0]
assert {"open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"} == c.keys()
assert c["high"] >= c["low"],  "high < low — невалидная свеча"
assert c["high"] >= c["open"] >= 0
assert c["volume"] >= 0
assert c["close"] > 0

print(f"✅ Binance клиент OK — {len(candles)} свечей BTCUSDT 15m")
print(f"   {c['open_time']} O={c['open']:.0f} H={c['high']:.0f} L={c['low']:.0f} C={c['close']:.0f} V={c['volume']:.1f}")
