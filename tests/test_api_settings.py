"""
tests/test_api_settings.py
Шаг 7: Тесты корректности структуры GET /api/settings и alias MAX_EDGE.
"""
import ast
import pathlib
from polyflip.settings_registry import registry_defaults

ROOT = pathlib.Path(__file__).parent.parent


# ── 7.3 Тест: DEFAULTS из сидера совпадают с реестром ─────────────────

def test_defaults_no_min_edge_matches_registry():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert DEFAULTS["NO_MIN_EDGE"] == registry_defaults()["NO_MIN_EDGE"]


def test_defaults_favorite_min_edge_matches_registry():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert DEFAULTS["FAVORITE_MIN_EDGE"] == registry_defaults()["FAVORITE_MIN_EDGE"]


def test_defaults_min_edge_matches_registry():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert DEFAULTS["MIN_EDGE"] == registry_defaults()["MIN_EDGE"]


def test_defaults_crypto_min_edge_matches_registry():
    from polyflip.db.init_runtime_settings import DEFAULTS
    assert DEFAULTS["CRYPTO_MIN_EDGE"] == registry_defaults()["CRYPTO_MIN_EDGE"]


# ── 7.4 Тест: legacy полей нет в Settings ────────────────────────────────

def test_no_legacy_fields_in_settings():
    from polyflip.config import settings
    legacy_keys = [
        "AUTO_DEAD_ZONE_WIDTH",
        "NO_MAX_PRICE",
        "TRADE_NO_MAX_PRICE",
    ]
    for key in legacy_keys:
        assert not hasattr(settings, key), f"Legacy поле {key} найдено в Settings!"
