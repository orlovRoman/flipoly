def test_defaults_include_edge_bounds():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert "MIN_EDGE" in DEFAULTS, "MIN_EDGE должен быть в дефолтах"
    assert "MAX_BET_EDGE" in DEFAULTS, "MAX_BET_EDGE должен быть в дефолтах"
    assert float(DEFAULTS["MIN_EDGE"]) > 0
    assert float(DEFAULTS["MAX_BET_EDGE"]) > float(DEFAULTS["MIN_EDGE"])

def test_edge_formula_is_roi_based():
    """Убедиться что новая формула edge это ROI, не разница вероятностей."""
    from polyflip.trading.position_sizing import compute_edge
    # win_prob=0.55, buy_price=0.50 → (0.55/0.50)-1 = 0.10, не 0.05
    edge = compute_edge(0.55, 0.50)
    assert abs(edge - 0.10) < 1e-3, f"edge должен быть ROI-based: 0.10, получили {edge}"
    assert abs(edge - 0.05) > 1e-3, "edge не должен быть старым линейным значением 0.05"

def test_defaults_max_edge_matches_constant():
    """DEFAULTS["MAX_BET_EDGE"] должен совпадать с constants.MAX_BET_EDGE_SCALING."""
    from polyflip.db.init_runtime_settings import DEFAULTS
    from polyflip.constants import MAX_BET_EDGE_SCALING as MAX_BET_EDGE
    assert float(DEFAULTS["MAX_BET_EDGE"]) == MAX_BET_EDGE, (
        f"DEFAULTS['MAX_BET_EDGE']={DEFAULTS['MAX_BET_EDGE']} "
        f"не совпадает с constants.MAX_BET_EDGE_SCALING={MAX_BET_EDGE}"
    )

def test_defaults_min_edge_matches_constant():
    from polyflip.db.init_runtime_settings import DEFAULTS
    from polyflip.constants import MIN_EDGE
    assert float(DEFAULTS["MIN_EDGE"]) == MIN_EDGE

def test_max_edge_is_higher_than_realistic_signals():
    """MAX_BET_EDGE должен быть > 0 и <= 1.0 (100% ROI = абсолютный максимум)."""
    from polyflip.constants import MAX_BET_EDGE_SCALING as MAX_BET_EDGE, MIN_EDGE
    assert MIN_EDGE < MAX_BET_EDGE
    assert MAX_BET_EDGE <= 1.0, "MAX_BET_EDGE > 100% ROI нереалистичен для Polymarket"
    assert MAX_BET_EDGE >= 0.10, "MAX_BET_EDGE слишком мал — будет срезать хорошие сигналы"

def test_max_edge_default_is_conservative():
    """MAX_BET_EDGE по умолчанию должен быть <= 0.25 (консервативный)"""
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert float(DEFAULTS["MAX_BET_EDGE"]) <= 0.25, (
        f"MAX_BET_EDGE={DEFAULTS['MAX_BET_EDGE']} слишком широкий — рискуем входить на неликвидных рынках"
    )

def test_max_edge_from_constants():
    from polyflip.constants import MAX_BET_EDGE_SCALING as MAX_BET_EDGE
    assert MAX_BET_EDGE <= 0.25
