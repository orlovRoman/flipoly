"""add_live_trading_sessions

Revision ID: e1f2a3b4c5d6
Revises: e7f8a9b0c1d2
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "e7f8a9b0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = set(inspector.get_table_names())

    if "live_trading_sessions" not in existing_tables:
        op.create_table(
            "live_trading_sessions",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "status", sa.String(length=24), nullable=False, server_default="DRAFT"
            ),
            sa.Column("budget_usdc", sa.Numeric(precision=18, scale=6), nullable=False),
            sa.Column(
                "reserved_usdc",
                sa.Numeric(precision=18, scale=6),
                nullable=False,
                server_default="0",
            ),
            sa.Column(
                "filled_usdc",
                sa.Numeric(precision=18, scale=6),
                nullable=False,
                server_default="0",
            ),
            sa.Column(
                "max_single_order_usdc",
                sa.Numeric(precision=18, scale=6),
                nullable=False,
            ),
            sa.Column(
                "max_total_exposure_usdc",
                sa.Numeric(precision=18, scale=6),
                nullable=False,
            ),
            sa.Column("max_open_positions", sa.Integer(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("stop_reason", sa.String(length=255), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.CheckConstraint("budget_usdc > 0", name="ck_live_session_budget_positive"),
            sa.CheckConstraint(
                "max_single_order_usdc > 0 AND max_single_order_usdc <= budget_usdc",
                name="ck_live_session_single_order",
            ),
            sa.CheckConstraint(
                "max_total_exposure_usdc > 0 AND max_total_exposure_usdc <= budget_usdc",
                name="ck_live_session_exposure",
            ),
            sa.CheckConstraint(
                "reserved_usdc >= 0 AND reserved_usdc <= budget_usdc",
                name="ck_live_session_reserved",
            ),
            sa.CheckConstraint(
                "max_open_positions BETWEEN 1 AND 100",
                name="ck_live_session_positions",
            ),
            sa.CheckConstraint(
                "status IN ('DRAFT','READY','ACTIVE','BUDGET_EXHAUSTED','STOPPED','ERROR')",
                name="ck_live_session_status",
            ),
        )

        single_active_predicate = sa.text("status IN ('DRAFT', 'READY', 'ACTIVE')")
        op.create_index(
            "uq_live_session_single_controllable",
            "live_trading_sessions",
            ["status"],
            unique=True,
            postgresql_where=single_active_predicate,
            sqlite_where=single_active_predicate,
        )

    exec_columns = {c["name"] for c in inspector.get_columns("execution_requests")}
    if "live_session_id" not in exec_columns:
        op.add_column(
            "execution_requests",
            sa.Column(
                "live_session_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("live_trading_sessions.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index(
            "ix_execution_requests_live_session",
            "execution_requests",
            ["live_session_id", "created_at"],
        )

    trade_columns = {c["name"] for c in inspector.get_columns("trade_history")}
    if "live_session_id" not in trade_columns:
        op.add_column(
            "trade_history",
            sa.Column(
                "live_session_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("live_trading_sessions.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index(
            "ix_trade_history_live_session",
            "trade_history",
            ["live_session_id", "position_status"],
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    trade_columns = {c["name"] for c in inspector.get_columns("trade_history")}
    if "live_session_id" in trade_columns:
        op.drop_index("ix_trade_history_live_session", table_name="trade_history")
        op.drop_column("trade_history", "live_session_id")

    exec_columns = {c["name"] for c in inspector.get_columns("execution_requests")}
    if "live_session_id" in exec_columns:
        op.drop_index("ix_execution_requests_live_session", table_name="execution_requests")
        op.drop_column("execution_requests", "live_session_id")

    existing_tables = set(inspector.get_table_names())
    if "live_trading_sessions" in existing_tables:
        op.drop_index(
            "uq_live_session_single_controllable", table_name="live_trading_sessions"
        )
        op.drop_table("live_trading_sessions")
