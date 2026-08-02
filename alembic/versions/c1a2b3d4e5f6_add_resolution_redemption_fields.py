"""add_resolution_redemption_fields

Revision ID: c1a2b3d4e5f6
Revises: b5e5f5g5h5i5
Create Date: 2026-08-02 02:04:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1a2b3d4e5f6'
down_revision: Union[str, None] = 'b5e5f5g5h5i5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "live_markets",
        sa.Column(
            "trading_status",
            sa.String(24),
            nullable=False,
            server_default="UNKNOWN",
        ),
    )
    op.add_column(
        "live_markets",
        sa.Column("accepting_orders", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "live_markets",
        sa.Column(
            "resolution_status",
            sa.String(24),
            nullable=False,
            server_default="PENDING",
        ),
    )
    op.add_column(
        "live_markets",
        sa.Column("final_outcome", sa.String(16), nullable=True),
    )
    op.add_column(
        "live_markets",
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "live_markets",
        sa.Column("resolution_checked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "live_markets",
        sa.Column("resolution_source", sa.String(32), nullable=True),
    )

    op.add_column(
        "trade_history",
        sa.Column("settlement_outcome", sa.String(16), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("expected_payout_usdc", sa.Numeric(38, 18), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("redeemable_shares", sa.Numeric(38, 18), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column(
            "redemption_status",
            sa.String(32),
            nullable=False,
            server_default="NOT_REQUIRED",
        ),
    )
    op.add_column(
        "trade_history",
        sa.Column("redemption_tx_hash", sa.String(128), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("redeemed_payout_usdc", sa.Numeric(38, 18), nullable=True),
    )
    op.add_column(
        "trade_history",
        sa.Column("redeemed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trade_history", "redeemed_at")
    op.drop_column("trade_history", "redeemed_payout_usdc")
    op.drop_column("trade_history", "redemption_tx_hash")
    op.drop_column("trade_history", "redemption_status")
    op.drop_column("trade_history", "redeemable_shares")
    op.drop_column("trade_history", "expected_payout_usdc")
    op.drop_column("trade_history", "settlement_outcome")

    op.drop_column("live_markets", "resolution_source")
    op.drop_column("live_markets", "resolution_checked_at")
    op.drop_column("live_markets", "resolved_at")
    op.drop_column("live_markets", "final_outcome")
    op.drop_column("live_markets", "resolution_status")
    op.drop_column("live_markets", "accepting_orders")
    op.drop_column("live_markets", "trading_status")
