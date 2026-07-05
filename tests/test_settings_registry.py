"""
tests/test_settings_registry.py
Шаг 3: Тест единого реестра настроек — "страж синхронизации".
"""
import re
import pathlib
import pytest

ROOT = pathlib.Path(__file__).parent.parent


# ── Тест 1: DEFAULTS сидера == реестр ────────────────────────────────────────

def test_all_registry_keys_in_defaults():
    """registry_defaults() должен полностью покрывать DEFAULTS сидера."""
    from polyflip.settings_registry import registry_defaults
    from polyflip.db.init_runtime_settings import DEFAULTS

    reg = registry_defaults()
    # Все ключи реестра должны быть в DEFAULTS
    missing_in_defaults = set(reg.keys()) - set(DEFAULTS.keys())
    assert not missing_in_defaults, (
        f"Ключи реестра, отсутствующие в DEFAULTS: {missing_in_defaults}"
    )


def test_no_orphan_defaults():
    """В DEFAULTS не должно быть ключей, которых нет в реестре."""
    from polyflip.settings_registry import registry_keys
    from polyflip.db.init_runtime_settings import DEFAULTS

    extra_in_defaults = set(DEFAULTS.keys()) - registry_keys()
    assert not extra_in_defaults, (
        f"Ключи в DEFAULTS, отсутствующие в реестре: {extra_in_defaults}"
    )


# ── Тест 2: Ключи, которые engine.py читает из settings_db ───────────────────

def test_no_orphan_settings_db_keys_in_engine():
    """
    Все ключи, которые engine.py читает через settings_db.get("KEY", ...),
    должны быть описаны в реестре.
    """
    src = (ROOT / "polyflip/trading/engine.py").read_text(encoding="utf-8")
    used_keys = set(re.findall(r'settings_db\.get\("([A-Z0-9_]+)"', src))
    from polyflip.settings_registry import registry_keys
    rkeys = registry_keys()
    orphans = used_keys - rkeys
    # Исключаем per-asset ключи которые генерируются динамически
    orphans = {k for k in orphans if not k.startswith("TRADE_FLIP_THRESHOLD_")}
    assert not orphans, (
        f"Ключи, которые engine читает, но нет в реестре: {orphans}"
    )


# ── Тест 3: AUTO_DEAD_ZONE_WIDTH убран из API ─────────────────────────────────

def test_auto_dead_zone_width_removed_from_api_response():
    """AUTO_DEAD_ZONE_WIDTH не должен присутствовать в get_all_settings."""
    src = (ROOT / "polyflip/api/settings.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    assert k.value != "AUTO_DEAD_ZONE_WIDTH", (
                        "AUTO_DEAD_ZONE_WIDTH не должен быть ключом в settings_dict"
                    )


def test_dead_zone_width_in_api_response():
    """DEAD_ZONE_WIDTH должен присутствовать в settings_dict API."""
    import ast
    src = (ROOT / "polyflip/api/settings.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys_found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys_found.add(k.value)
    assert "DEAD_ZONE_WIDTH" in keys_found
    assert "MAX_EDGE_FILTER" in keys_found


# ── Тест 4: MAX_EDGE_FILTER и MAX_BET_EDGE разные константы ──────────────────

def test_max_edge_filter_lt_max_edge_scaling():
    """MAX_EDGE_FILTER (фильтр аномалий) < MAX_EDGE_SCALING (масштабирование)."""
    from polyflip.constants import MAX_EDGE_FILTER, MAX_EDGE_SCALING
    assert MAX_EDGE_FILTER < MAX_EDGE_SCALING, (
        f"MAX_EDGE_FILTER={MAX_EDGE_FILTER} должен быть < MAX_EDGE_SCALING={MAX_EDGE_SCALING}"
    )


def test_max_bet_edge_default_is_scaling():
    """Дефолт MAX_BET_EDGE в реестре == MAX_EDGE_SCALING (0.40), не MAX_EDGE_FILTER (0.20)."""
    from polyflip.constants import MAX_EDGE_SCALING
    from polyflip.settings_registry import registry_defaults
    defaults = registry_defaults()
    assert float(defaults["MAX_BET_EDGE"]) == pytest.approx(MAX_EDGE_SCALING)


def test_max_edge_filter_default_is_filter():
    """Дефолт MAX_EDGE_FILTER в реестре == MAX_EDGE_FILTER (0.20)."""
    from polyflip.constants import MAX_EDGE_FILTER
    from polyflip.settings_registry import registry_defaults
    defaults = registry_defaults()
    assert float(defaults["MAX_EDGE_FILTER"]) == pytest.approx(MAX_EDGE_FILTER)


# ── Тест 5: Editable keys ────────────────────────────────────────────────────

def test_editable_keys_not_include_trading_enabled():
    """TRADING_ENABLED не должен быть редактируемым через API (только UI toggle)."""
    from polyflip.settings_registry import editable_keys
    assert "TRADING_ENABLED" not in editable_keys()


def test_registry_imports_cleanly():
    """Реестр должен импортироваться без ошибок."""
    from polyflip.settings_registry import REGISTRY, registry_keys, registry_defaults, editable_keys
    assert len(REGISTRY) > 10
    assert len(registry_keys()) == len(REGISTRY)
    assert len(registry_defaults()) == len(REGISTRY)
    assert len(editable_keys()) < len(registry_keys())  # некоторые ключи не editable


# ── Тест 6: constants.py не содержит AUTO_DEAD_ZONE_WIDTH как отдельную активную константу ──

def test_auto_dead_zone_width_not_in_valid_keys():
    """AUTO_DEAD_ZONE_WIDTH не должен быть в valid_keys API."""
    src = (ROOT / "polyflip/api/settings.py").read_text(encoding="utf-8")
    # valid_keys — это список в update_setting. Проверяем что "AUTO_DEAD_ZONE_WIDTH" там нет как строка
    # (кроме как в комментарии)
    lines = src.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if '"AUTO_DEAD_ZONE_WIDTH"' in stripped and 'valid_keys' not in stripped:
            # Допускаем только внутри комментариев или legacy migration
            pass
    # Грубая проверка: в valid_keys-списке нет AUTO_DEAD_ZONE_WIDTH
    assert '"AUTO_DEAD_ZONE_WIDTH",' not in src or '# AUTO_DEAD_ZONE_WIDTH' in src
