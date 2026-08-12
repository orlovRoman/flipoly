"""add execution policy and quote telemetry"""
from alembic import op
import sqlalchemy as sa

revision = "20260813_execution_telemetry"
down_revision = "20260812_live_market_strike"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("execution_requests") as batch_op:
        batch_op.add_column(sa.Column("execution_order_mode", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("post_only", sa.Boolean(), server_default=sa.text("false"), nullable=False))
        batch_op.add_column(sa.Column("decision_price", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("release_quote_price", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("release_quote_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("submit_quote_price", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("submit_quote_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("submitted_limit_price", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("cancel_due_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("terminal_code", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("network_retry_count", sa.Integer(), server_default=sa.text("0"), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("execution_requests") as batch_op:
        for name in (
            "network_retry_count", "terminal_code", "cancel_due_at",
            "submitted_limit_price", "submit_quote_at", "submit_quote_price",
            "release_quote_at", "release_quote_price", "decision_price",
            "post_only", "execution_order_mode",
        ):
            batch_op.drop_column(name)