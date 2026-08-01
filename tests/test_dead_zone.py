import pytest
from polyflip.trading.utils import compute_dead_zone

def test_compute_dead_zone_auto():
    lower, upper = compute_dead_zone(0.70, 0.10, auto_mode=True)
    assert lower == 0.65
    assert upper == 0.75

def test_compute_dead_zone_manual():
    lower, upper = compute_dead_zone(0.70, 0.10, auto_mode=False)
    assert lower == 0.60
    assert upper == 0.70

def test_dead_zone_width_consistency():
    lower, upper = compute_dead_zone(0.65, 0.12, auto_mode=True)
    assert round(upper - lower, 4) == 0.12
