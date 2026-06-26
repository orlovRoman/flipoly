import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from polyflip.db.models import Base

@pytest_asyncio.fixture(scope="session")
async def engine():
    engine_obj = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    async with engine_obj.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine_obj
    await engine_obj.dispose()

@pytest_asyncio.fixture(scope="function")
async def db_session(engine):
    async with engine.connect() as conn:
        await conn.begin()
        await conn.begin_nested()  # SAVEPOINT
        session = AsyncSession(bind=conn, expire_on_commit=False)
        yield session
        await session.close()
        await conn.rollback()
