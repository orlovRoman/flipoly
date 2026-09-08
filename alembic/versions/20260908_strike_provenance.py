"""Add strike provenance columns to live_markets and market_snapshots.

Revision ID: 20260908_strike_provenance
Revises: 20260908_volume_status
"""
from alembic import op
import sqlalchemy as sa

revision = "20260908_strike_provenance"
down_revision = "20260908_volume_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # live_markets
    op.add_column("live_markets", sa.Column("strike_value", sa.Float(), nullable=True))
    op.add_column("live_markets", sa.Column("strike_source", sa.String(length=64), nullable=True))
    op.add_column("live_markets", sa.Column("strike_effective_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("live_markets", sa.Column("strike_received_at", sa.DateTime(timezone=True), nullable=True))

    # market_snapshots
    op.add_column("market_snapshots", sa.Column("strike_value", sa.Float(), nullable=True))
    op.add_column("market_snapshots", sa.Column("strike_source", sa.String(length=64), nullable=True))
    op.add_column("market_snapshots", sa.Column("strike_effective_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("market_snapshots", sa.Column("strike_received_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # market_snapshots
    op.drop_column("market_snapshots", "strike_received_at")
    op.drop_column("market_snapshots", "strike_effective_at")
    op.drop_column("market_snapshots", "strike_source")
    op.drop_column("market_snapshots", "strike_value")

    # live_markets
    op.drop_column("live_markets", "strike_received_at")
    op.drop_column("live_markets", "strike_effective_at")
    op.drop_column("live_markets", "strike_source")
    op.drop_column("live_markets", "strike_value")
