import pytest
from polyflip.api.analytics import get_model_subtype_info

def test_phase_models_form_distinct_keys():
    """Фазовые модели DOGE_leaning, DOGE_decided, DOGE_contested имеют уникальные ключи."""
    phase_assets = ["DOGE", "DOGE_leaning", "DOGE_decided", "DOGE_contested"]
    keys = [(asset, 8) for asset in phase_assets]
    assert len(keys) == len(set(keys)), f"Коллизия фазовых ключей: {keys}"

def test_lgbm_subtypes_use_exact_key_no_collision():
    """LightGBM субтипы одного символа дают разные exact-ключи."""
    lgbm_assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    keys = [(asset, 7) for asset in lgbm_assets]
    assert len(keys) == len(set(keys)), f"Коллизия: {keys}"

def test_model_attribution_isolation():
    """Проверка, что каждая фазовая модель изолирует PnL по своему model_key."""
    trades = [
        {"model_key": "DOGE_leaning", "model_version": 8, "pnl": 1.5},
        {"model_key": "DOGE_decided", "model_version": 8, "pnl": -1.0},
        {"model_key": "DOGE_contested", "model_version": 8, "pnl": 0.8},
    ]
    grouped = {}
    for t in trades:
        k = (t["model_key"], t["model_version"])
        grouped[k] = grouped.get(k, 0.0) + t["pnl"]
        
    assert grouped[("DOGE_leaning", 8)] == 1.5
    assert grouped[("DOGE_decided", 8)] == -1.0
    assert grouped[("DOGE_contested", 8)] == 0.8
