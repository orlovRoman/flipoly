"""Add orderbook_depth_snapshots table and market bounds / strike observed columns.

Revision ID: 20260909_orderbook_depth_and_market_bounds
Revises: 20260908_underlying_observations
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260909_orderbook_depth_and_market_bounds"
down_revision = "20260908_underlying_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create orderbook_depth_snapshots table
    op.create_table(
        "orderbook_depth_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("market_snapshots.id"), nullable=True),
        sa.Column("market_id", sa.String(length=128), nullable=False),
        sa.Column("token_id", sa.String(length=128), nullable=False),
        sa.Column("outcome_side", sa.String(length=16), nullable=False),
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("bids", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=False),
        sa.Column("asks", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=False),
        sa.Column("sequence_id", sa.String(length=64), nullable=True),
        sa.Column("is_truncated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("depth_limit", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="CLOB"),
        sa.Column("quality_status", sa.String(length=32), nullable=False, server_default="VALID"),
        sa.Column("quality_notes", sa.String(length=256), nullable=True),
        sa.Column("best_bid_price", sa.Float(), nullable=True),
        sa.Column("best_bid_size", sa.Float(), nullable=True),
        sa.Column("best_ask_price", sa.Float(), nullable=True),
        sa.Column("best_ask_size", sa.Float(), nullable=True),
        sa.Column("depth_usdc_bid", sa.Float(), nullable=True),
        sa.Column("depth_usdc_ask", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_orderbook_market_token", "orderbook_depth_snapshots", ["market_id", "token_id", "received_at"])
    op.create_index("idx_orderbook_received_side", "orderbook_depth_snapshots", ["market_id", "outcome_side", "received_at"])
    op.create_index("idx_orderbook_snapshot_id", "orderbook_depth_snapshots", ["snapshot_id"])

    # 2. Add market bounds and strike observed columns to market_snapshots
    op.add_column("market_snapshots", sa.Column("strike_observed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("market_snapshots", sa.Column("market_start_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("market_snapshots", sa.Column("market_end_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("market_snapshots", sa.Column("settlement_price_source", sa.String(length=64), nullable=True))

    # 3. Add market bounds and strike observed columns to live_markets
    op.add_column("live_markets", sa.Column("strike_observed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("live_markets", sa.Column("market_start_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("live_markets", sa.Column("market_end_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("live_markets", sa.Column("settlement_price_source", sa.String(length=64), nullable=True))


def downgrade() -> None:
    # live_markets
    op.drop_column("live_markets", "settlement_price_source")
    op.drop_column("live_markets", "market_end_at")
    op.drop_column("live_markets", "market_start_at")
    op.drop_column("live_markets", "strike_observed_at")

    # market_snapshots
    op.drop_column("market_snapshots", "settlement_price_source")
    op.drop_column("market_snapshots", "market_end_at")
    op.drop_column("market_snapshots", "market_start_at")
    op.drop_column("market_snapshots", "strike_observed_at")

    # orderbook_depth_snapshots
    op.drop_index("idx_orderbook_snapshot_id", table_name="orderbook_depth_snapshots")
    op.drop_index("idx_orderbook_received_side", table_name="orderbook_depth_snapshots")
    op.drop_index("idx_orderbook_market_token", table_name="orderbook_depth_snapshots")
    op.drop_table("orderbook_depth_snapshots")
