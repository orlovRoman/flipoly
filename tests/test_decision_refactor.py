import pytest
from polyflip.trading.decision_runners import _get_float_setting
from polyflip.trading.feature_builder import MarketSignal
from polyflip.trading.decision_logic import decide_favorite


def test_get_float_setting_no_division():
    # Нормальное значение — не должно делиться
    assert _get_float_setting({"FLIP_THRESHOLD": "0.8"}, "FLIP_THRESHOLD") == 0.8

    # Значение > 1.0 — теперь возвращается как есть (не делится на 100)
    assert _get_float_setting({"FLIP_THRESHOLD": "80"}, "FLIP_THRESHOLD") == 80.0

    # Пустые и нулевые — None
    assert _get_float_setting({"FLIP_THRESHOLD": ""}, "FLIP_THRESHOLD") is None
    assert _get_float_setting({"FLIP_THRESHOLD": "0"}, "FLIP_THRESHOLD") is None
    assert _get_float_setting({"FLIP_THRESHOLD": "0.0"}, "FLIP_THRESHOLD") is None
    assert _get_float_setting({}, "FLIP_THRESHOLD") is None


def _signal(mid: float, spread: float = 0.02) -> MarketSignal:
    """
    MarketSignal автоматически вычисляет yes_ask и no_ask свойства на основе mid_price и spread:
    yes_ask = min(mid_price + spread / 2, 0.99)
    no_ask = min((1.0 - mid_price) + spread / 2, 0.99)
    """
    return MarketSignal(
        asset="TEST",
        mid_price=mid,
        spread=spread,
        volume_5min=500.0,
        price_velocity=0.0,
        hour_of_day=12,
        time_left_min=60.0,
    )


_CFG = {
    "FAVORITE_THRESHOLD": "0.40",
    "FAVORITE_MIN_PRICE": "0.55",
    "FAVORITE_MAX_PRICE": "0.95",
    "DEAD_ZONE_WIDTH":    "0.05",
    "FAVORITE_MIN_EDGE":  "-1.0",
    "TRADE_BET_SIZE_USDC": "5.0",
    "BET_SIZING_MODE": "fixed",
}


def test_no_side_not_blocked_by_yes_out_of_bounds():
    """
    Был баг: YES out-of-bounds → SKIP, NO-side не проверялась.
    При mid=0.42, spread=0.02:
      yes_ask = 0.43 (< fav_min 0.55 — out of bounds)
      no_ask = 0.59 (в границах [0.55, 0.95])
    Ранее функция блокировалась на YES-side. Теперь должна вернуть BUY_NO.
    """
    signal = _signal(mid=0.42, spread=0.02)
    d = decide_favorite(signal, _CFG)
    assert d.action == "BUY_NO", f"Expected BUY_NO, got {d.action}: {d.reason}"


def test_yes_side_works_normally():
    """YES-side должна работать при нормальных условиях."""
    signal = _signal(mid=0.72, spread=0.02)
    d = decide_favorite(signal, _CFG)
    assert d.action == "BUY_YES", f"Expected BUY_YES, got {d.action}: {d.reason}"


def test_dead_zone_skip():
    """mid_price=0.50 в мёртвой зоне → SKIP."""
    signal = _signal(mid=0.50, spread=0.02)
    d = decide_favorite(signal, _CFG)
    assert d.action == "SKIP" and "dead zone" in d.reason


def test_both_sides_prefers_higher_edge():
    """
    Если обе стороны в пределах допустимых цен (в bounds) — выбирается с бо́льшим edge.
    При mid=0.42, spread=0.34:
      yes_ask = 0.42 + 0.17 = 0.59 (в bounds) -> edge = 0.42/0.59 - 1 = -0.288
      no_ask = 0.58 + 0.17 = 0.75 (в bounds)  -> edge = 0.58/0.75 - 1 = -0.226
    NO edge (-0.226) выше, чем YES edge (-0.288), поэтому выбирается BUY_NO.
    """
    signal = _signal(mid=0.42, spread=0.34)
    d = decide_favorite(signal, _CFG)
    assert d.action == "BUY_NO", f"Expected BUY_NO (higher edge), got {d.action}"
