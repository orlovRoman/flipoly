import os

import pytest
from sqlalchemy import text


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_execution_fill_idempotency_constraint_exists_in_postgres():
    if os.getenv("POSTGRES_INTEGRATION_TESTS") != "1":
        pytest.skip("PostgreSQL integration checks are disabled")

    from polyflip.db.connection import engine

    assert engine.dialect.name == "postgresql"
    async with engine.connect() as connection:
        definition = (await connection.execute(text("""
                    SELECT pg_get_constraintdef(constraint_row.oid)
                    FROM pg_constraint AS constraint_row
                    JOIN pg_class AS table_row
                      ON table_row.oid = constraint_row.conrelid
                    WHERE table_row.relname = 'execution_fills'
                      AND constraint_row.conname = 'uq_execution_provider_trade'
                      AND constraint_row.contype = 'u'
                    """))).scalar_one_or_none()

    assert definition is not None
    normalized = " ".join(definition.lower().split())
    assert "unique (gateway, provider_trade_id)" in normalized
