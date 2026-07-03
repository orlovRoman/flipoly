import json
import urllib.request
import urllib.error

url = "http://89.223.120.30:8000/api/backtest/submit"
headers = {"Content-Type": "application/json"}
payload = {
    "assets": ["BTC", "ETH"],
    "date_from": None,
    "date_to": None,
    "min_snapshots_per_market": 3,
    "model_id": None,
    "strategy_mode": "ML",
    "min_time_left_min": 1,
    "max_time_left_min": 60,
    "no_flip_threshold": 0.35,
    "flip_threshold": 0.60,
    "trade_on_flip": False,
    "favorite_threshold": 0.65,
    "auto_dead_zone_width": 0.10,
    "yes_min_price": 0.55,
    "yes_max_price": 0.95,
    "no_min_price": 0.55,
    "no_max_price": 0.95,
    "initial_capital": 1000,
    "bet_sizing_mode": "scaled",
    "trade_bet_size_usdc": 5,
    "max_bet_size_usdc": 50,
    "min_edge": -0.05,
    "max_edge": 0.50,
    "slippage_pct": 0.005
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers=headers, method='POST')

try:
    with urllib.request.urlopen(req) as f:
        print("SUCCESS")
        print(f.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP ERROR", e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print("OTHER ERROR", e)
