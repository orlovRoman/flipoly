"""Add underlying_observations table and oracle_price/binance_price columns.

Revision ID: 20260908_underlying_observations
Revises: 20260908_strike_provenance
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260908_underlying_observations"
down_revision = "20260908_strike_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create underlying_observations table
    op.create_table(
        "underlying_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("instrument", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("extra_data", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("instrument", "source", "event_at", "received_at", name="uix_underlying_obs_dedup"),
    )
    op.create_index("idx_underlying_obs_lookup", "underlying_observations", ["instrument", "source", "event_at"])
    op.create_index("idx_underlying_obs_received", "underlying_observations", ["instrument", "received_at"])

    # 2. Add oracle_price and binance_price columns to live_markets
    op.add_column("live_markets", sa.Column("oracle_price", sa.Float(), nullable=True))
    op.add_column("live_markets", sa.Column("binance_price", sa.Float(), nullable=True))

    # 3. Add oracle_price and binance_price columns to market_snapshots
    op.add_column("market_snapshots", sa.Column("oracle_price", sa.Float(), nullable=True))
    op.add_column("market_snapshots", sa.Column("binance_price", sa.Float(), nullable=True))


def downgrade() -> None:
    # market_snapshots
    op.drop_column("market_snapshots", "binance_price")
    op.drop_column("market_snapshots", "oracle_price")

    # live_markets
    op.drop_column("live_markets", "binance_price")
    op.drop_column("live_markets", "oracle_price")

    # underlying_observations
    op.drop_index("idx_underlying_obs_received", table_name="underlying_observations")
    op.drop_index("idx_underlying_obs_lookup", table_name="underlying_observations")
    op.drop_table("underlying_observations")
