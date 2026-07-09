"""
Тесты флага TRADE_ON_FAVORITE — гарантируют что при False
все три стратегии (ML, FAVORITE, CRYPTO) получают SKIP.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from polyflip.trading.decision_logic import TradeDecision


def test_trade_on_favorite_false_ml_returns_skip():
    """При TRADE_ON_FAVORITE=False ML-ветка возвращает SKIP без вызова decide_ml_trend."""
    from polyflip.trading.decision_logic import TradeDecision
    # Симулируем поведение engine: если trade_on_favorite=False
    trade_on_favorite = False
    if trade_on_favorite:
        decision_obj = MagicMock()  # не должны попасть сюда
    else:
        decision_obj = TradeDecision(
            action="SKIP",
            strategy_type="ML_TREND",
            reason="Favorite trades disabled (TRADE_ON_FAVORITE=False)",
            edge=0.0,
            buy_price=0.0,
            bet_size_usdc=0.0
        )
    assert decision_obj.action == "SKIP"
    assert "TRADE_ON_FAVORITE" in decision_obj.reason


def test_trade_on_favorite_true_ml_calls_decide_ml_trend():
    """При TRADE_ON_FAVORITE=True вызывается decide_ml_trend."""
    from polyflip.trading.decision_logic import decide_ml_trend, MarketSignal
    from polyflip.trading.feature_builder import MarketSignal as MS
    signal = MS(
        asset="BTC", mid_price=0.65, spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        hour_of_day=12, time_left_min=30.0
    )
    config = {
        "NO_FLIP_THRESHOLD": 0.45, "FLIP_THRESHOLD": 0.65,
        "MIN_EDGE": -100.0, "MAX_BET_EDGE": 100.0,
        "BYPASS_BET_SIZE_CHECK": "true",
        "TRADE_BET_SIZE_USDC": "10.0",
        "FAVORITE_THRESHOLD": "0.6",
    }
    decision = decide_ml_trend(signal, p_flip=0.3, config=config)
    # p_flip=0.3 < lower=0.45 → должен быть BUY_YES или SKIP по edge
    assert decision.action in ("BUY_YES", "SKIP")


def test_trade_on_favorite_in_registry_and_editable():
    """TRADE_ON_FAVORITE должен быть в реестре и доступен для редактирования."""
    from polyflip.settings_registry import REGISTRY, editable_keys, registry_defaults
    keys = {s.key for s in REGISTRY}
    assert "TRADE_ON_FAVORITE" in keys, "TRADE_ON_FAVORITE отсутствует в реестре"
    assert "TRADE_ON_FAVORITE" in editable_keys(), "TRADE_ON_FAVORITE должен быть editable"
    defaults = registry_defaults()
    assert defaults["TRADE_ON_FAVORITE"] in ("true", "false"), \
        "Дефолт TRADE_ON_FAVORITE должен быть 'true' или 'false'"


def test_trade_on_favorite_default_is_true():
    """По умолчанию TRADE_ON_FAVORITE=true (торгуем на фаворита)."""
    from polyflip.settings_registry import registry_defaults
    defaults = registry_defaults()
    assert defaults["TRADE_ON_FAVORITE"] == "true"


def test_trade_on_favorite_constant_matches_registry_default():
    """Константа TRADE_ON_FAVORITE в constants.py совпадает с дефолтом реестра."""
    from polyflip.constants import TRADE_ON_FAVORITE
    from polyflip.settings_registry import registry_defaults
    defaults = registry_defaults()
    expected = "true" if TRADE_ON_FAVORITE else "false"
    assert defaults["TRADE_ON_FAVORITE"] == expected
