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


def _signal(mid, yes_ask, no_ask) -> MarketSignal:
    return MarketSignal(
        asset="TEST",
        mid_price=mid,
        spread=abs(yes_ask - no_ask),
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
    """Был баг: YES out-of-bounds → SKIP, NO-side не проверялась."""
    signal = _signal(mid=0.42, yes_ask=0.43, no_ask=0.62)  # yes_ask < fav_min=0.55
    d = decide_favorite(signal, _CFG)
    assert d.action == "BUY_NO", f"Expected BUY_NO, got {d.action}: {d.reason}"


def test_yes_side_works_normally():
    """YES-side должна работать при нормальных условиях."""
    signal = _signal(mid=0.72, yes_ask=0.73, no_ask=0.28)
    d = decide_favorite(signal, _CFG)
    assert d.action == "BUY_YES", f"Expected BUY_YES, got {d.action}: {d.reason}"


def test_dead_zone_skip():
    """mid_price=0.50 в мёртвой зоне → SKIP."""
    signal = _signal(mid=0.50, yes_ask=0.51, no_ask=0.49)
    d = decide_favorite(signal, _CFG)
    assert d.action == "SKIP" and "dead zone" in d.reason


def test_both_sides_prefers_higher_edge():
    """Если обе стороны подходят — выбирается с бо́льшим edge."""
    signal = _signal(mid=0.42, yes_ask=0.57, no_ask=0.62)
    d = decide_favorite(signal, _CFG)
    assert d.action == "BUY_NO", f"Expected BUY_NO (higher edge), got {d.action}"
