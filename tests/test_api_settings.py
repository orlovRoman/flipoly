"""
tests/test_api_settings.py
Шаг 7: Тесты корректности структуры GET /api/settings и alias MAX_EDGE.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


# ── 7.1 Тест: ключи в GET /api/settings (AST-анализ settings.py) ──────────

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

def test_no_min_edge_constant_value():
    """
    Значение в константах должно быть явно задокументировано.
    Ранее сидер сеял '0.05', а константа была 0.04.
    При смене нужно учитывать, что у старых инсталляций в БД останется старое значение.
    """
    from polyflip.constants import NO_MIN_EDGE
    import pytest
    assert NO_MIN_EDGE == pytest.approx(0.04), "Если меняешь — обновить и DEFAULTS и комментарий"
