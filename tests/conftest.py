import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from polyflip.db.models import Base
import polyflip.db.execution_models  # Ensure execution models are registered
import pytest
from polyflip.trading.ml_inference import clear_models_cache


from contextlib import contextmanager


@contextmanager
def override_dependencies(app, overrides: dict):
    """Safely override FastAPI dependencies and restore previous state."""
    original = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    try:
        yield
    finally:
        for k, v in original.items():
            if v is None:
                app.dependency_overrides.pop(k, None)
            else:
                app.dependency_overrides[k] = v


@pytest.fixture(autouse=True)
def clean_models_cache_fixture():
    clear_models_cache()


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


@pytest_asyncio.fixture
async def pg_session_factory():
    """Фикстура для подключения к PostgreSQL при прогоне postgres-тестов."""
    import os
    from sqlalchemy import text

    required = os.getenv("POSTGRES_INTEGRATION_TESTS") == "1"
    pg_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")

    if not pg_url:
        if required:
            pytest.fail("PostgreSQL integration DB URL is required")
        pytest.skip("PostgreSQL integration disabled")

    engine = create_async_engine(pg_url, pool_pre_ping=True)

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        if required:
            pytest.fail(f"PostgreSQL integration DB unavailable: {exc}")
        pytest.skip(str(exc))

    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


from polyflip.trading.schemas import TradeExecution, ExecutionFees
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4


def make_dummy_execution(
    status="FILLED",
    mode="LIVE",
    executed_price=0.5,
    executed_usdc=5.0,
    filled_shares=10.0,
    error_msg=None,
    attempt_id=None,
):
    if attempt_id is None:
        attempt_id = uuid4()
    fees = ExecutionFees(
        platform_fee_usdc=Decimal("0.0"),
        builder_fee_usdc=Decimal("0.0"),
        network_fee_native=Decimal("0.0"),
        network_fee_symbol="POL",
        network_fee_usdc=Decimal("0.0"),
        fee_source="CONFIRMED_ZERO",
    )
    return TradeExecution(
        attempt_id=attempt_id,
        provider_order_id="dummy_order",
        provider_status="FILLED",
        status="PAPER_FILLED" if mode == "PAPER" else "FILLED",
        side="BUY",
        order_type="FOK",
        token_id="dummy_token",
        original_requested_shares=Decimal(str(filled_shares)),
        submitted_shares=Decimal(str(filled_shares)),
        filled_shares=Decimal(str(filled_shares)),
        net_position_delta_shares=Decimal(str(filled_shares)),
        average_price=Decimal(str(executed_price)),
        gross_quote_usdc=Decimal(str(executed_usdc)),
        net_quote_usdc=Decimal(str(executed_usdc)),
        liquidity_role="UNKNOWN",
        fees=fees,
        trade_ids=("t1",),
        transaction_hashes=("h1",),
        submitted_at=datetime.now(timezone.utc),
        observed_at=datetime.now(timezone.utc),
        error_message=error_msg,
    )
