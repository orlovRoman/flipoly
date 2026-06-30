# tests/test_backtest_schemas.py
import pytest
from datetime import datetime, timezone
from polyflip.api.backtest_schemas import BacktestConfig, BacktestResult

def test_default_config_valid():
    cfg = BacktestConfig()
    assert cfg.strategy_mode == "ML"
    assert cfg.assets == ["BTC", "ETH"]

def test_to_runner_config_keys():
    cfg = BacktestConfig()
    d = cfg.to_runner_config()
    required_keys = [
        "MIN_TIME_LEFT_MIN", "MAX_TIME_LEFT_MIN", "STRATEGY_MODE",
        "FAVORITE_THRESHOLD", "KELLY_ENABLED", "KELLY_MULTIPLIER",
        "NO_FLIP_THRESHOLD", "FLIP_THRESHOLD"
    ]
    for key in required_keys:
        assert key in d, f"Missing key: {key}"

def test_max_time_validator():
    with pytest.raises(Exception):
        BacktestConfig(min_time_left_min=30.0, max_time_left_min=10.0)

def test_custom_config_roundtrip():
    cfg = BacktestConfig(
        assets=["BTC"],
        favorite_threshold=0.70,
        kelly_multiplier=0.5,
        initial_capital=5000.0
    )
    d = cfg.to_runner_config()
    assert d["FAVORITE_THRESHOLD"] == 0.70
    assert d["KELLY_MULTIPLIER"] == 0.5
    assert d["INITIAL_CAPITAL"] == 5000.0
