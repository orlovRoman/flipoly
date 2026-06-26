import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from polyflip.db.models import Base

@pytest_asyncio.fixture(scope="function")
async def db_session():
    # SQLite in-memory database for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    TestingSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    async with TestingSessionLocal() as session:
        yield session
        
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
    await engine.dispose()
