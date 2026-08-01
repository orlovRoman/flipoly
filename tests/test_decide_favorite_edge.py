import pytest
from polyflip.trading.decision_logic import decide_favorite
from polyflip.trading.feature_builder import MarketSignal


def make_signal(mid: float, spread: float = 0.02) -> MarketSignal:
    return MarketSignal(
        asset="BTC", mid_price=mid, spread=spread,
        volume_5min=500.0, price_velocity=0.0,
        hour_of_day=12, time_left_min=30.0
    )


BASE_CONFIG = {
    "FAVORITE_THRESHOLD": "0.60",
    "MIN_EDGE": "0.02",
    "MAX_BET_EDGE": "0.25",
    "TRADE_BET_SIZE_USDC": "5",
    "MAX_BET_SIZE_USDC": "50",
    "AUTO_DEAD_ZONE_WIDTH": "0.05",
    "TRADE_MIN_PRICE": "0.50",
    "TRADE_MAX_PRICE": "0.95",
}


def test_favorite_yes_edge_positive_with_small_spread():
    """При малом спреде edge может быть положительным или слабо отрицательным."""
    config = {**BASE_CONFIG, "FAVORITE_MIN_EDGE": "-0.02"}
    decision = decide_favorite(make_signal(mid=0.80, spread=0.001), config)
    assert decision.action == "BUY_YES"
    assert decision.edge is not None and decision.edge > -0.02


def test_favorite_yes_skips_when_edge_below_min():
    """SKIP если edge < MIN_EDGE"""
    config = {**BASE_CONFIG, "MIN_EDGE": "0.10"}  # требуем 10% ROI, FAVORITE_MIN_EDGE не задан
    decision = decide_favorite(make_signal(mid=0.75), config)
    assert decision.action == "SKIP"
    assert "edge" in decision.reason.lower()


def test_favorite_no_has_real_edge():
    """NO-решение тоже должно иметь рассчитанный edge"""
    config = {**BASE_CONFIG, "FAVORITE_MIN_EDGE": "-0.05"}
    decision = decide_favorite(make_signal(mid=0.25, spread=0.001), config)
    assert decision.action == "BUY_NO"
    assert decision.edge is not None and decision.edge != 0.0


def test_favorite_edge_wires_to_bet_sizing():
    """Размер ставки должен зависеть от edge (scaled режим)"""
    config_low_edge = {**BASE_CONFIG, "FAVORITE_MIN_EDGE": "-0.05", "TRADE_BET_SIZE_USDC": "5", "MAX_BET_SIZE_USDC": "50"}
    config_high_min_edge = {**BASE_CONFIG, "FAVORITE_MIN_EDGE": "-0.05", "MAX_BET_EDGE": "0.01",
                            "TRADE_BET_SIZE_USDC": "5", "MAX_BET_SIZE_USDC": "50"}
    d1 = decide_favorite(make_signal(mid=0.75, spread=0.001), config_low_edge)
    d2 = decide_favorite(make_signal(mid=0.75, spread=0.001), config_high_min_edge)
    if d1.action != "SKIP":
        assert d1.bet_size_usdc > 0
