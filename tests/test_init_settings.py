def test_defaults_include_edge_bounds():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert "MIN_EDGE" in DEFAULTS, "MIN_EDGE должен быть в дефолтах"
    assert "MAX_EDGE" in DEFAULTS, "MAX_EDGE должен быть в дефолтах"
    assert float(DEFAULTS["MIN_EDGE"]) > 0
    assert float(DEFAULTS["MAX_EDGE"]) > float(DEFAULTS["MIN_EDGE"])

def test_edge_formula_is_roi_based():
    """Убедиться что новая формула edge это ROI, не разница вероятностей."""
    from polyflip.trading.position_sizing import compute_edge
    # win_prob=0.55, buy_price=0.50 → (0.55/0.50)-1 = 0.10, не 0.05
    edge = compute_edge(0.55, 0.50)
    assert abs(edge - 0.10) < 1e-3, f"edge должен быть ROI-based: 0.10, получили {edge}"
    assert abs(edge - 0.05) > 1e-3, "edge не должен быть старым линейным значением 0.05"
