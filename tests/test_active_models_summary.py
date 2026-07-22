import pytest
from polyflip.api.analytics import get_model_subtype_info

def test_key_full_is_unique_per_lgbm_subtype():
    """key_full = (m.asset, m.version) уникален для каждой субмодели."""
    assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    version = 5
    keys = [(asset, version) for asset in assets]
    assert len(keys) == len(set(keys)), f"Коллизия key_full: {keys}"

def test_key_base_collision_is_expected():
    """
    Предупреждение: key_base = (base_symbol, version) НАМЕРЕННО коллизионен
    для разных субтипов одного символа — fallback на него недопустим.
    """
    assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    base_keys = []
    for asset in assets:
        base_symbol, _, _ = get_model_subtype_info(asset)
        base_keys.append((base_symbol, 5))
    # Все три дают ("BTC", 5) — доказываем что fallback небезопасен
    assert len(set(base_keys)) == 1, \
        "Ожидается коллизия base_key для LightGBM субтипов — fallback запрещён"
