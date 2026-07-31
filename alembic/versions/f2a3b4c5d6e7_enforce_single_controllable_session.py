"""enforce_single_controllable_session

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Добавляем колонку network_chain_id в execution_worker_status если нет
    ws_columns = {c["name"] for c in inspector.get_columns("execution_worker_status")}
    if "network_chain_id" not in ws_columns:
        op.add_column(
            "execution_worker_status",
            sa.Column("network_chain_id", sa.Integer(), nullable=True),
        )

    # Пересоздаем частичный уникальный индекс на выражение ((1)), перекрывающий ВСЕ 3 статуса
    existing_indexes = {
        idx["name"] for idx in inspector.get_indexes("live_trading_sessions")
    }
    if "uq_live_session_single_controllable" in existing_indexes:
        op.drop_index(
            "uq_live_session_single_controllable", table_name="live_trading_sessions"
        )

    dialect = connection.dialect.name
    if dialect == "postgresql":
        op.execute(
            """
            CREATE UNIQUE INDEX uq_live_session_single_controllable 
            ON live_trading_sessions ((1)) 
            WHERE status IN ('DRAFT', 'READY', 'ACTIVE')
            """
        )
    else:
        # SQLite
        op.execute(
            """
            CREATE UNIQUE INDEX uq_live_session_single_controllable 
            ON live_trading_sessions (status) 
            WHERE status IN ('DRAFT', 'READY', 'ACTIVE')
            """
        )


def downgrade() -> None:
    connection = op.get_bind()
    dialect = connection.dialect.name
    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS uq_live_session_single_controllable")
    else:
        op.drop_index(
            "uq_live_session_single_controllable", table_name="live_trading_sessions"
        )

    inspector = sa.inspect(connection)
    ws_columns = {c["name"] for c in inspector.get_columns("execution_worker_status")}
    if "network_chain_id" in ws_columns:
        op.drop_column("execution_worker_status", "network_chain_id")
