"""Create the atomic reservation table used by BTC_CT_T5_V1.

Revision ID: 20260910_ct_decision_reservations
Revises: 20260909_snapshot_mid_price_spread_nullable
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260910_ct_decision_reservations"
down_revision = "20260909_snapshot_mid_price_spread_nullable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ct_decision_reservations",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("market_id", sa.String(length=128), nullable=False),
        sa.Column("spec_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("decision_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("side", sa.String(length=16), nullable=True),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("budget_usdc", sa.Float(), nullable=True),
        sa.Column("reason", sa.String(length=128), nullable=False),
        sa.Column("trade_history_id", sa.Integer(), nullable=True),
        sa.Column(
            "decision_details",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=True,
        ),
        sa.Column("repeat_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_repeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_index(
        "idx_ct_decision_reservations_market_id",
        "ct_decision_reservations",
        ["market_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_ct_decision_reservations_market_id",
        table_name="ct_decision_reservations",
    )
    op.drop_table("ct_decision_reservations")
