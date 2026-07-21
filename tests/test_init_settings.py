from polyflip.settings_registry import registry_defaults

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

def test_defaults_max_edge_matches_registry():
    """DEFAULTS["MAX_BET_EDGE"] должен совпадать с реестром."""
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert float(DEFAULTS["MAX_BET_EDGE"]) == float(registry_defaults()["MAX_BET_EDGE"])

def test_defaults_min_edge_matches_registry():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert float(DEFAULTS["MIN_EDGE"]) == float(registry_defaults()["MIN_EDGE"])

def test_max_edge_is_higher_than_realistic_signals():
    """MAX_BET_EDGE должен быть > 0 и <= 1.0 (100% ROI = абсолютный максимум)."""
    defaults = registry_defaults()
    min_e = float(defaults["MIN_EDGE"])
    max_e = float(defaults["MAX_BET_EDGE"])
    assert min_e < max_e
    assert max_e <= 1.0, "MAX_BET_EDGE > 100% ROI нереалистичен для Polymarket"
    assert max_e >= 0.10, "MAX_BET_EDGE слишком мал — будет срезать хорошие сигналы"

def test_max_edge_default_is_conservative():
    """MAX_BET_EDGE по умолчанию должен быть <= 0.40"""
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert float(DEFAULTS["MAX_BET_EDGE"]) <= 0.40, (
        f"MAX_BET_EDGE={DEFAULTS['MAX_BET_EDGE']} слишком широкий — рискуем входить на неликвидных рынках"
    )

def test_max_edge_from_registry():
    max_bet_edge = float(registry_defaults()["MAX_BET_EDGE"])
    assert max_bet_edge <= 0.40
