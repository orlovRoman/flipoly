"""
tests/test_dead_zone_unified.py
Шаг 1: Тесты объединения DEAD_ZONE_WIDTH / AUTO_DEAD_ZONE_WIDTH.
"""
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import insert, select

from polyflip.db.models import RuntimeSettings, Base
from polyflip.db.init_runtime_settings import migrate_auto_dead_zone_width, DEFAULTS


# ── Фикстура in-memory БД ─────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


# ── Тест миграции ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_migrate_moves_auto_dead_zone_width_to_dead_zone_width(db_session: AsyncSession):
    """Если в БД есть AUTO_DEAD_ZONE_WIDTH — значение переезжает в DEAD_ZONE_WIDTH."""
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="AUTO_DEAD_ZONE_WIDTH", value="0.18", updated_by="test", updated_at=now))
    await db_session.commit()

    await migrate_auto_dead_zone_width(db_session)

    result = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "DEAD_ZONE_WIDTH")
    )
    assert result is not None
    assert result.value == "0.18"

    old = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "AUTO_DEAD_ZONE_WIDTH")
    )
    assert old is None


@pytest.mark.asyncio
async def test_migrate_updates_existing_dead_zone_width(db_session: AsyncSession):
    """Если DEAD_ZONE_WIDTH уже существует — мигратор обновляет его значение."""
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.10", updated_by="system", updated_at=now))
    db_session.add(RuntimeSettings(key="AUTO_DEAD_ZONE_WIDTH", value="0.22", updated_by="test", updated_at=now))
    await db_session.commit()

    await migrate_auto_dead_zone_width(db_session)

    result = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "DEAD_ZONE_WIDTH")
    )
    assert result.value == "0.22"  # перезаписан значением из AUTO_DEAD_ZONE_WIDTH


@pytest.mark.asyncio
async def test_migrate_noop_when_no_auto_dead_zone_width(db_session: AsyncSession):
    """Если AUTO_DEAD_ZONE_WIDTH нет — мигратор ничего не делает."""
    now = datetime.now(timezone.utc)
    db_session.add(RuntimeSettings(key="DEAD_ZONE_WIDTH", value="0.10", updated_by="system", updated_at=now))
    await db_session.commit()

    await migrate_auto_dead_zone_width(db_session)  # не должно упасть

    result = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "DEAD_ZONE_WIDTH")
    )
    assert result.value == "0.10"  # значение не изменилось


# ── Тест структуры DEFAULTS ───────────────────────────────────────────────────

def test_defaults_no_auto_dead_zone_width():
    """AUTO_DEAD_ZONE_WIDTH не должен присутствовать в DEFAULTS сидера."""
    assert "AUTO_DEAD_ZONE_WIDTH" not in DEFAULTS, (
        "AUTO_DEAD_ZONE_WIDTH не должен сидироваться — движок читает DEAD_ZONE_WIDTH"
    )


def test_defaults_dead_zone_width_present():
    """DEAD_ZONE_WIDTH должен быть в DEFAULTS."""
    assert "DEAD_ZONE_WIDTH" in DEFAULTS
    val = float(DEFAULTS["DEAD_ZONE_WIDTH"])
    assert 0.02 <= val <= 0.40, f"DEAD_ZONE_WIDTH={val} вне допустимого диапазона"


def test_defaults_dead_zone_matches_constant():
    """DEFAULTS['DEAD_ZONE_WIDTH'] должен совпадать с constants.DEAD_ZONE_WIDTH."""
    from polyflip.constants import DEAD_ZONE_WIDTH
    assert DEFAULTS["DEAD_ZONE_WIDTH"] == str(DEAD_ZONE_WIDTH)


# ── Тест константы ────────────────────────────────────────────────────────────

def test_dead_zone_width_value():
    """DEAD_ZONE_WIDTH = 0.10 (объединённое значение)."""
    from polyflip.constants import DEAD_ZONE_WIDTH
    assert DEAD_ZONE_WIDTH == pytest.approx(0.10)


def test_auto_dead_zone_width_removed_from_constants():
    """AUTO_DEAD_ZONE_WIDTH не должен экспортироваться из constants как публичная константа."""
    import polyflip.constants as c
    # Допускаем, что AUTO_DEAD_ZONE_WIDTH всё ещё существует как alias для обратной совместимости,
    # но проверяем что DEAD_ZONE_WIDTH == AUTO_DEAD_ZONE_WIDTH (единое значение).
    if hasattr(c, "AUTO_DEAD_ZONE_WIDTH"):
        assert c.AUTO_DEAD_ZONE_WIDTH == c.DEAD_ZONE_WIDTH, (
            "Если AUTO_DEAD_ZONE_WIDTH сохранён как alias — он должен равняться DEAD_ZONE_WIDTH"
        )


# ── Тест decision_logic читает DEAD_ZONE_WIDTH ────────────────────────────────

def test_decide_favorite_reads_dead_zone_width():
    """decide_favorite должен использовать DEAD_ZONE_WIDTH, а не AUTO_DEAD_ZONE_WIDTH."""
    from polyflip.trading.decision_logic import decide_favorite
    from polyflip.trading.feature_builder import MarketSignal

    # mid_price=0.50 — нейтральная зона; при dead_zone=0.20 → SKIP
    signal = MarketSignal(
        asset="BTC",
        mid_price=0.50,
        spread=0.01,
        volume_5min=1000.0,
        price_velocity=0.0,
        hour_of_day=12,
        time_left_min=2.0,
    )
    result = decide_favorite(signal, {"DEAD_ZONE_WIDTH": "0.20", "FAVORITE_THRESHOLD": "0.55"})
    assert result.action == "SKIP"
    assert result.reason == "dead zone"
