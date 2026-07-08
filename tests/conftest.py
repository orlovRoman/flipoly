import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from polyflip.db.models import Base


@pytest_asyncio.fixture(scope="function")
async def engine():
    """Отдельный in-memory движок на каждый тест — полная изоляция."""
    engine_obj = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine_obj.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine_obj
    async with engine_obj.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine_obj.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(engine):
    """Чистая сессия на каждый тест. Без SAVEPOINT — просто commit/rollback."""
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
