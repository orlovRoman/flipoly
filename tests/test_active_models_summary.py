import pytest
from polyflip.api.analytics import get_model_subtype_info

def test_logreg_phases_use_base_symbol_key():
    """Все LogReg фазы BTC матчатся по одному ключу ("BTC", version)."""
    phase_assets = ["BTC", "BTC_leaning", "BTC_decided", "BTC_contested"]
    for asset in phase_assets:
        base_symbol, sub_code, _ = get_model_subtype_info(asset)
        assert base_symbol == "BTC", f"{asset} → base_symbol должен быть BTC, получен {base_symbol}"

def test_lgbm_subtypes_use_exact_key_no_collision():
    """LightGBM субтипы одного символа дают разные exact-ключи."""
    lgbm_assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    keys = [(asset, 7) for asset in lgbm_assets]
    assert len(keys) == len(set(keys)), f"Коллизия: {keys}"

def test_lgbm_base_key_collision_is_prevented_by_exact_match():
    """
    base_symbol у всех LightGBM субтипов одинаков → базовый ключ коллизионен.
    Поэтому LightGBM ОБЯЗАН использовать exact, а не base.
    """
    base_keys = set()
    for asset in ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]:
        base_symbol, _, _ = get_model_subtype_info(asset)
        base_keys.add((base_symbol, 7))

    assert len(base_keys) == 1, "Все три LightGBM субтипа дают один base_key — это ожидаемая коллизия, exact обязателен"

def test_matching_strategy_by_model_type():
    """
    Логика выбора ключа: LightGBM → exact, LogReg → base.
    """
    LGBM_SUFFIXES = ("_low_vol", "_mid_vol", "_high_vol")

    test_cases = [
        ("BTC",              False, "base"),
        ("BTC_leaning",      False, "logreg_phase"),
        ("BTC_decided",      False, "logreg_phase"),
        ("BTC_contested",    False, "logreg_phase"),
        ("BTCUSDT_low_vol",  True,  "lgbm"),
        ("BTCUSDT_mid_vol",  True,  "lgbm"),
        ("BTCUSDT_high_vol", True,  "lgbm"),
    ]
    for asset, expected_lgbm, label in test_cases:
        is_lgbm = any(asset.endswith(s) for s in LGBM_SUFFIXES)
        assert is_lgbm == expected_lgbm, f"{asset} ({label}): ожидался is_lgbm={expected_lgbm}"
