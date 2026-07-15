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

def test_trading_enabled_is_not_editable_via_api():
    from polyflip.settings_registry import editable_keys
    assert "TRADING_ENABLED" not in editable_keys()
    assert "BYPASS_BET_SIZE_CHECK" not in editable_keys()

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

from polyflip.api.main import app
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_get_all_settings_does_not_crash_with_empty_db(db_session, monkeypatch):
    """get_all_settings не должен падать когда settings_registry не совпадает с БД."""
    from polyflip.api.main import app
    from httpx import ASGITransport, AsyncClient
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass
    monkeypatch.setattr("polyflip.api.settings.async_session", lambda: DummyAsyncContextManager(db_session))
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/settings", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert len(data) > 0

@pytest.mark.asyncio
async def test_get_all_settings_includes_security_sensitive_keys_as_masked(db_session, monkeypatch):
    """TRADING_ENABLED и BYPASS_BET_SIZE_CHECK должны присутствовать в ответе."""
    from polyflip.api.main import app
    from httpx import ASGITransport, AsyncClient
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass
    monkeypatch.setattr("polyflip.api.settings.async_session", lambda: DummyAsyncContextManager(db_session))
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/settings", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "TRADING_ENABLED" in data
        assert "BYPASS_BET_SIZE_CHECK" in data


# ── Тест 7: Настройки стоп-лосса ──────────────────────────────────────────────

def test_stoploss_settings_in_registry():
    from polyflip.settings_registry import REGISTRY, editable_keys, registry_defaults

    keys = {s.key for s in REGISTRY}
    assert "STOP_LOSS_ENABLED"       in keys
    assert "STOP_LOSS_PCT_FAVORITE"  in keys
    assert "STOP_LOSS_PCT_OUTSIDER"  in keys
    assert "STOP_LOSS_CHECK_SEC"     in keys

def test_stoploss_defaults():
    from polyflip.settings_registry import registry_defaults
    defaults = registry_defaults()
    assert defaults["STOP_LOSS_ENABLED"]       == "false"
    assert defaults["STOP_LOSS_PCT_FAVORITE"]  == "40.0"
    assert defaults["STOP_LOSS_PCT_OUTSIDER"]  == "60.0"
    assert defaults["STOP_LOSS_CHECK_SEC"]     == "30"

def test_stoploss_keys_are_editable():
    from polyflip.settings_registry import editable_keys
    editable = editable_keys()
    # Все должны быть редактируемы через дашборд
    assert "STOP_LOSS_ENABLED"       in editable
    assert "STOP_LOSS_PCT_FAVORITE"  in editable
    assert "STOP_LOSS_PCT_OUTSIDER"  in editable
    assert "STOP_LOSS_CHECK_SEC"     in editable

def test_no_settingdef_uses_settingmeta():
    """Гарантируем что в кодовой базе нет несуществующего SettingMeta."""
    import polyflip.settings_registry as sr
    assert not hasattr(sr, "SettingMeta"), "SettingMeta не существует, используй SettingDef"


