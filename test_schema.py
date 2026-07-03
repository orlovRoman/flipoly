import json
from polyflip.api.backtest_schemas import BacktestConfig
from pydantic import ValidationError

payload = {
    "assets": [],
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

try:
    config = BacktestConfig(**payload)
    print("SUCCESS")
except ValidationError as e:
    print("VALIDATION ERROR")
    print(e.errors())
