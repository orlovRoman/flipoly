import pytest
from polyflip.settings_registry import registry_defaults

def test_defaults_include_edge_bounds():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert "OUTS_MIN_EDGE" in DEFAULTS, "OUTS_MIN_EDGE должен быть в дефолтах"
    assert float(DEFAULTS["OUTS_MIN_EDGE"]) > 0

def test_edge_formula_is_roi_based():
    from polyflip.trading.position_sizing import compute_edge
    # win_prob=0.55, buy_price=0.50 -> 0.05
    edge = compute_edge(0.55, 0.50)
    assert abs(edge - 0.05) < 1e-3, f"edge must be 0.05, got {edge}"

def test_defaults_min_edge_matches_registry():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert float(DEFAULTS["OUTS_MIN_EDGE"]) == float(registry_defaults()["OUTS_MIN_EDGE"])
