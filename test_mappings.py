import asyncio
from polyflip.crypto.binance_client import COIN_TO_SYMBOL, fetch_klines
from polyflip.crypto.candle_collector import SYMBOLS
from polyflip.crypto.historical_loader import DEFAULT_SYMBOLS

def test_mappings():
    assert "DOGEUSDT" in COIN_TO_SYMBOL.values()
    assert "XRPUSDT"  in COIN_TO_SYMBOL.values()
    assert "SOLUSDT"  in COIN_TO_SYMBOL.values()
    
    candles = fetch_klines("DOGEUSDT", "15m", limit=5)
    assert len(candles) == 5
    assert candles[0]["close"] > 0
    print("Binance client mapping and fetch OK")

    assert set(["DOGEUSDT", "XRPUSDT", "SOLUSDT"]).issubset(set(SYMBOLS))
    print("Candle collector SYMBOLS OK")

    assert "DOGEUSDT" in DEFAULT_SYMBOLS
    print("Historical loader DEFAULT_SYMBOLS OK")

if __name__ == "__main__":
    test_mappings()
