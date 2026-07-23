import pytest
import structlog
from polyflip.trading.decision_logic import (
    decide_favorite, decide_ml_trend, decide_outsider, TradeDecision
)
from polyflip.trading.feature_builder import MarketSignal
from polyflip.constants import ECE_WARN_THRESHOLD

def test_step2_ml_trend_edge_uses_ml_probability():
    """Шаг 2: Edge в decide_ml_trend должен считаться через ML вероятности (p_flip)"""
    signal = MarketSignal(
        asset="BTC", mid_price=0.52, yes_ask=0.53, no_ask=0.49,
        spread=0.01, volume_5min=100.0, price_velocity=0.0,
        hour_of_day=12, time_left_min=10.0
    )
    config = {
        "NO_FLIP_THRESHOLD": 0.35, "MIN_EDGE": 0.05,
        "FAVORITE_MIN_PRICE": 0.50, "FAVORITE_MAX_PRICE": 0.95,
        "DEAD_ZONE_WIDTH": 0.05
    }
    # p_flip = 0.1 -> p_win = 0.9 -> edge = 0.9 / 0.53 - 1 ≈ 0.6981
    decision = decide_ml_trend(signal, p_flip=0.1, config=config, ece=0.0)
    assert decision.action == "BUY_YES"
    assert decision.edge is not None
    expected_edge = 0.9 / 0.53 - 1.0
    assert abs(decision.edge - expected_edge) < 0.001

def test_step3_outsider_threshold_uses_calibrated_p_flip():
    """Шаг 3: Порог FLIP_THRESHOLD в decide_outsider должен сравниваться с откалиброванным p_flip"""
    signal = MarketSignal(
        asset="BTC", mid_price=0.70, yes_ask=0.72, no_ask=0.32,
        spread=0.01, volume_5min=100.0, price_velocity=0.0,
        hour_of_day=12, time_left_min=10.0
    )
    config = {
        "FLIP_THRESHOLD": 0.60, "NO_MIN_EDGE": 0.04, "MIN_EDGE": 0.04,
        "OUTSIDER_MAX_PRICE": 0.45, "DEAD_ZONE_WIDTH": 0.05
    }
    # p_flip=0.61 (raw), ECE=0.10 -> calibrated = 0.5 + (0.61 - 0.5) * (1 - 0.10) = 0.599 < 0.60 -> SKIP
    decision = decide_outsider(signal, p_flip=0.61, config=config, ece=0.10)
    assert decision.action == "SKIP"
    assert "p_flip_calibrated" in decision.reason

def test_step4_outsider_direction():
    """Шаг 4: Рефактор decide_outsider — проверка правильности вычисления результирующих действий"""
    # YES фаворит (mid_price 0.70) -> покупаем NO
    signal_yes_fav = MarketSignal(
        asset="BTC", mid_price=0.70, yes_ask=0.72, no_ask=0.32,
        spread=0.01, volume_5min=100.0, price_velocity=0.0,
        hour_of_day=12, time_left_min=10.0
    )
    config = {
        "FLIP_THRESHOLD": 0.60, "NO_MIN_EDGE": 0.01, "MIN_EDGE": 0.01,
        "OUTSIDER_MAX_PRICE": 0.45, "DEAD_ZONE_WIDTH": 0.05
    }
    decision = decide_outsider(signal_yes_fav, p_flip=0.70, config=config, ece=0.0)
    assert decision.action == "BUY_NO"
    assert decision.buy_price == 0.32

def test_step5_ece_warn_threshold_constant():
    """Шаг 5: Проверка экспорта константы ECE_WARN_THRESHOLD"""
    assert ECE_WARN_THRESHOLD == 0.07

def test_step6_favorite_threshold_parsing():
    """Шаг 6: Проверка логики разбора FAVORITE_THRESHOLD"""
    signal = MarketSignal(
        asset="BTC", mid_price=0.50, yes_ask=0.51, no_ask=0.51,
        spread=0.01, volume_5min=100.0, price_velocity=0.0,
        hour_of_day=12, time_left_min=10.0
    )
    # 0.50 находится в dead zone (0.10) -> SKIP
    decision = decide_favorite(signal, {"FAVORITE_THRESHOLD": "invalid"})
    assert decision.action == "SKIP"
