import pytest
from polyflip.api.analytics import get_model_subtype_info

def test_no_key_collision_for_lgbm_subtypes():
    """Три LightGBM-субмодели одного символа не должны давать одинаковый ключ."""
    assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    keys = []
    for asset in assets:
        base_symbol, sub_code, _ = get_model_subtype_info(asset)
        keys.append((asset, 1))  # version=1 у всех одинаковый

    assert len(keys) == len(set(keys)), f"Коллизия ключей: {keys}"
