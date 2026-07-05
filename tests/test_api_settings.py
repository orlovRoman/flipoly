"""
tests/test_api_settings.py
Шаг 7: Тесты корректности структуры GET /api/settings и alias MAX_EDGE.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


# ── 7.1 Тест: ключи в GET /api/settings (AST-анализ settings.py) ──────────

def _get_settings_dict_keys() -> set[str]:
    """Парсит AST polyflip/api/settings.py и возвращает ключи settings_dict."""
    src = (ROOT / "polyflip/api/settings.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    return keys


def test_settings_api_no_duplicate_max_edge():
    """MAX_EDGE не должен появляться в GET /api/settings — только MAX_BET_EDGE."""
    keys = _get_settings_dict_keys()
    assert "MAX_EDGE" not in keys, "MAX_EDGE не должен присутствовать в GET /api/settings"
    assert "MAX_BET_EDGE" in keys, "MAX_BET_EDGE должен присутствовать в GET /api/settings"


def test_settings_api_edge_fields_present():
    """FAVORITE_MIN_EDGE, NO_MIN_EDGE, CRYPTO_MIN_EDGE должны быть в GET /api/settings."""
    keys = _get_settings_dict_keys()
    for expected in ("FAVORITE_MIN_EDGE", "NO_MIN_EDGE", "CRYPTO_MIN_EDGE"):
        assert expected in keys, f"{expected} отсутствует в GET /api/settings"


def test_settings_api_no_duplicate_max_bet_size():
    """MAX_BET_SIZE_USDC должен встречаться как ключ ровно 1 раз в словаре."""
    src = (ROOT / "polyflip/api/settings.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    count = 0
    # Ищем в первом Dict, который явно является settings_dict (его ключ ACTIVE_FEATURES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            node_keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "ACTIVE_FEATURES" in node_keys:
                count = node_keys.count("MAX_BET_SIZE_USDC")
                break
    assert count == 1, f"MAX_BET_SIZE_USDC встречается {count} раз в settings_dict (ожидается 1)"


# ── 7.3 Тест: DEFAULTS из сидера совпадают с константами ─────────────────

def test_defaults_no_min_edge_matches_constant():
    """NO_MIN_EDGE в DEFAULTS должен равняться значению из constants.py."""
    from polyflip.db.init_runtime_settings import DEFAULTS
    from polyflip.constants import NO_MIN_EDGE
    assert DEFAULTS["NO_MIN_EDGE"] == str(NO_MIN_EDGE), (
        f"Расхождение: DEFAULTS['NO_MIN_EDGE']={DEFAULTS['NO_MIN_EDGE']}, "
        f"constants.NO_MIN_EDGE={NO_MIN_EDGE}"
    )


def test_defaults_favorite_min_edge_matches_constant():
    from polyflip.db.init_runtime_settings import DEFAULTS
    from polyflip.constants import FAVORITE_MIN_EDGE
    assert DEFAULTS["FAVORITE_MIN_EDGE"] == str(FAVORITE_MIN_EDGE)


def test_defaults_min_edge_matches_constant():
    from polyflip.db.init_runtime_settings import DEFAULTS
    from polyflip.constants import MIN_EDGE
    assert DEFAULTS["MIN_EDGE"] == str(MIN_EDGE)


def test_defaults_crypto_min_edge_matches_constant():
    from polyflip.db.init_runtime_settings import DEFAULTS
    from polyflip.constants import CRYPTO_MIN_EDGE
    assert DEFAULTS["CRYPTO_MIN_EDGE"] == str(CRYPTO_MIN_EDGE)


# ── 7.4 Тест: legacy полей нет в Settings ────────────────────────────────

def test_no_legacy_fields_in_settings():
    """TRADE_ONLY_FAVORITE, DRIFT_THRESHOLD, BET_FRACTION не должны быть в Settings."""
    from polyflip.config import settings
    for field in ("TRADE_ONLY_FAVORITE", "DRIFT_THRESHOLD", "BET_FRACTION"):
        assert not hasattr(settings, field), (
            f"Legacy-поле {field} всё ещё присутствует в Settings"
        )


# ── Тест: engine.py не ссылается на legacy-поля из settings ──────────────

def test_engine_no_references_to_removed_legacy():
    """engine.py не должен обращаться к удалённым legacy-полям settings."""
    src = (ROOT / "polyflip/trading/engine.py").read_text(encoding="utf-8")
    for field in ("TRADE_ONLY_FAVORITE", "DRIFT_THRESHOLD", "BET_FRACTION"):
        assert field not in src, f"engine.py содержит ссылку на удалённое поле {field}"
