import pytest
from polyflip.trading.feature_builder import MarketSignal
from polyflip.trading.decision_logic import decide_favorite, decide_ml_trend, decide_outsider

BASE_CONFIG = {
    "FAVORITE_THRESHOLD": 0.65,
    "MIN_EDGE": -0.05,
    "MAX_EDGE": 0.20,
    "YES_MIN_PRICE": 0.55, "YES_MAX_PRICE": 0.95,
    "NO_MIN_PRICE": 0.55,  "NO_MAX_PRICE": 0.95,
    "AUTO_DEAD_ZONE_WIDTH": 0.10,
    "INITIAL_CAPITAL": 1000.0,
    "TRADE_BET_SIZE_USDC": 5.0,
    "MAX_BET_SIZE_USDC": 50.0,
    "NO_FLIP_THRESHOLD": 0.35,
    "FLIP_THRESHOLD": 0.60,
    "OUTSIDER_NO_MIN_PRICE": 0.10,
    "OUTSIDER_NO_MAX_PRICE": 0.50,
}

def make_signal(**kwargs):
    defaults = dict(
        asset="BTC", mid_price=0.70, spread=0.02,
        volume_5min=5000.0, price_velocity=0.01,
        hour_of_day=12, time_left_min=5.0
    )
    defaults.update(kwargs)
    return MarketSignal(**defaults)

# --- PURE_FAVORITE ---
def test_favorite_buys_yes_when_high():
    d = decide_favorite(make_signal(mid_price=0.75), BASE_CONFIG)
    assert d.action == "BUY_YES"
    assert d.strategy_type == "PURE_FAVORITE"
    assert d.bet_size_usdc > 0

def test_favorite_buys_no_when_low():
    d = decide_favorite(make_signal(mid_price=0.25), BASE_CONFIG)
    assert d.action == "BUY_NO"

def test_favorite_skips_dead_zone():
    d = decide_favorite(make_signal(mid_price=0.50), BASE_CONFIG)
    assert d.action == "SKIP"

def test_favorite_skips_on_low_edge():
    # spread очень маленький, но цена в диапазоне — edge может быть OK
    # Тест: цена вне диапазона → SKIP
    cfg = {**BASE_CONFIG, "YES_MAX_PRICE": 0.60}
    d = decide_favorite(make_signal(mid_price=0.80), cfg)
    assert d.action == "SKIP"

def test_favorite_bet_within_limits():
    d = decide_favorite(make_signal(mid_price=0.75), BASE_CONFIG)
    assert BASE_CONFIG["TRADE_BET_SIZE_USDC"] <= d.bet_size_usdc <= BASE_CONFIG["MAX_BET_SIZE_USDC"]

# --- ML_TREND ---
def test_ml_trend_skips_high_p_flip():
    d = decide_ml_trend(make_signal(mid_price=0.75), p_flip=0.70, config=BASE_CONFIG)
    assert d.action == "SKIP"
    assert d.p_flip == 0.70

def test_ml_trend_buys_when_low_p_flip():
    d = decide_ml_trend(make_signal(mid_price=0.75), p_flip=0.20, config=BASE_CONFIG)
    assert d.action == "BUY_YES"
    assert d.strategy_type == "ML_TREND"
    assert d.p_flip == 0.20

# --- OUTSIDER ---
def test_outsider_buys_no_when_yes_is_favorite():
    d = decide_outsider(make_signal(mid_price=0.75), p_flip=0.80, config=BASE_CONFIG)
    assert d.action == "BUY_NO"
    assert d.strategy_type == "OUTSIDER"

def test_outsider_skips_low_p_flip():
    d = decide_outsider(make_signal(mid_price=0.75), p_flip=0.30, config=BASE_CONFIG)
    assert d.action == "SKIP"

def test_all_decisions_have_reason():
    signals = [make_signal(mid_price=p) for p in [0.25, 0.50, 0.75]]
    for sig in signals:
        for decision in [
            decide_favorite(sig, BASE_CONFIG),
            decide_ml_trend(sig, 0.5, BASE_CONFIG),
            decide_outsider(sig, 0.5, BASE_CONFIG),
        ]:
            assert isinstance(decision.reason, str) and len(decision.reason) > 0

def test_outsider_buys_yes_when_no_is_favorite():
    cfg = {**BASE_CONFIG, "OUTSIDER_YES_MIN_PRICE": 0.05, "OUTSIDER_YES_MAX_PRICE": 0.45}
    d = decide_outsider(make_signal(mid_price=0.25), p_flip=0.80, config=cfg)
    assert d.action == "BUY_YES"
    assert d.strategy_type == "OUTSIDER"

def test_outsider_yes_rejects_overpriced():
    cfg = {**BASE_CONFIG, "OUTSIDER_YES_MIN_PRICE": 0.05, "OUTSIDER_YES_MAX_PRICE": 0.20}
    d = decide_outsider(make_signal(mid_price=0.25, spread=0.02), p_flip=0.80, config=cfg)
    assert d.action == "SKIP"

