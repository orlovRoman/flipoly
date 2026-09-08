"""Add volume_status column to live_markets and market_snapshots.

Revision ID: 20260908_volume_status
Revises: 20260905_merge_weighted_policy_heads
"""
from alembic import op
import sqlalchemy as sa

revision = "20260908_volume_status"
down_revision = "20260905_merge_weighted_policy_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "live_markets",
        sa.Column("volume_status", sa.String(length=32), nullable=True, server_default="VALID"),
    )
    op.add_column(
        "market_snapshots",
        sa.Column("volume_status", sa.String(length=32), nullable=True, server_default="VALID"),
    )


def downgrade() -> None:
    op.drop_column("market_snapshots", "volume_status")
    op.drop_column("live_markets", "volume_status")
