import numpy as np
import pytest
from polyflip.trading.feature_builder import (
    MarketSignal, build_feature_vector, FEATURE_COLUMNS
)

def make_signal(**kwargs):
    defaults = dict(
        asset="BTC", mid_price=0.7, spread=0.02,
        volume_5min=5000.0, price_velocity=0.01,
        hour_of_day=12, time_left_min=5.0
    )
    defaults.update(kwargs)
    return MarketSignal(**defaults)

def test_feature_vector_shape():
    signal = make_signal()
    vec = build_feature_vector(signal)
    assert vec.shape == (1, len(FEATURE_COLUMNS))

def test_feature_vector_order():
    signal = make_signal(mid_price=0.65, time_left_min=3.0)
    vec = build_feature_vector(signal)
    # time_left_min — первый элемент
    assert vec[0][0] == 3.0
    # mid_price — второй
    assert vec[0][1] == 0.65

def test_yes_ask_capped_at_099():
    signal = make_signal(mid_price=0.98, spread=0.05)
    assert signal.yes_ask == 0.99

def test_no_ask_floored():
    signal = make_signal(mid_price=0.99, spread=0.001)
    assert signal.no_ask <= 0.99
    assert signal.no_bid >= 0.01

def test_spread_zero_fallback():
    signal = make_signal(spread=0.0)
    assert signal.yes_ask == signal.yes_bid == signal.mid_price


def test_build_feature_vector_no_nan():
    from polyflip.trading.feature_builder import build_feature_vector, FEATURE_COLUMNS
    from polyflip.trading.decision_logic import MarketSignal
    signal = MarketSignal(
        asset="BTC", mid_price=0.65, spread=0.02,
        volume_5min=100.0, price_velocity=0.01,
        hour_of_day=14, time_left_min=10.0
    )
    vec = build_feature_vector(signal)
    import numpy as np
    assert not np.any(np.isnan(vec)), \
        f"NaN в feature vector на позициях: {[FEATURE_COLUMNS[i] for i in np.where(np.isnan(vec[0]))[0]]}"


def test_get_price_phase_importable_and_works():
    from polyflip.constants import get_price_phase
    assert get_price_phase(0.15) in ("contested", "leaning", "decided")
    assert get_price_phase(0.50) is not None
    assert get_price_phase(0.85) is not None

