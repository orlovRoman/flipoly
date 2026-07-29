import pytest
from polyflip.execution.worker import _get_dialect, _acquire_mode_lock

@pytest.mark.asyncio
async def test_get_dialect_does_not_raise(db_session):
    """_get_dialect не должен выбрасывать AttributeError при async_sessionmaker."""
    try:
        dialect = await _get_dialect(db_session)
        assert dialect == "sqlite"
    except AttributeError as e:
        pytest.fail(f"_get_dialect кинул AttributeError: {e}")

@pytest.mark.asyncio
async def test_acquire_mode_lock_sqlite_noop(db_session):
    """На SQLite _acquire_mode_lock должен проходить как no-op без ошибок."""
    await _acquire_mode_lock(db_session, "LIVE")
    await _acquire_mode_lock(db_session, "PAPER")
