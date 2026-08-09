"""add market_direction_signals

Revision ID: c3d4e5f6a7b9
Revises: b2c3d4e5f6a8
Create Date: 2026-08-09 16:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "c3d4e5f6a7b9"
down_revision = "b2c3d4e5f6a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_direction_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("market_id", sa.String(length=128), nullable=False),
        sa.Column("asset", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("p_up", sa.Float(), nullable=False),
        sa.Column("p_down", sa.Float(), nullable=False),
        sa.Column("signal_strength", sa.Float(), nullable=False),
        sa.Column("strike", sa.Float(), nullable=True),
        sa.Column("threshold_up", sa.Float(), nullable=False),
        sa.Column("threshold_down", sa.Float(), nullable=False),
        sa.Column("model_key", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("features_ok", sa.Boolean(), nullable=False),
        sa.Column("risk_vetoed", sa.Boolean(), nullable=False),
        sa.Column("risk_reason", sa.String(length=256), nullable=True),
        sa.Column("stake_multiplier", sa.Float(), nullable=False),
        sa.Column("funding_rate", sa.Float(), nullable=False),
        sa.Column("ece", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("inverted", sa.Boolean(), nullable=False),
        sa.Column("p_up_raw", sa.Float(), nullable=False),
        sa.Column("p_down_raw", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("market_id", name="uq_market_direction_signals_market_id"),
    )
    op.create_index(
        "idx_direction_signal_asset_created",
        "market_direction_signals",
        ["asset", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_direction_signal_market",
        "market_direction_signals",
        ["market_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_direction_signal_market", table_name="market_direction_signals")
    op.drop_index(
        "idx_direction_signal_asset_created",
        table_name="market_direction_signals",
    )
    op.drop_table("market_direction_signals")
