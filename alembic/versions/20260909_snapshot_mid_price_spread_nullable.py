"""Make market_snapshots mid_price and spread nullable.

Revision ID: 20260909_snapshot_mid_price_spread_nullable
Revises: 20260909_orderbook_depth_and_market_bounds
"""
from alembic import op
import sqlalchemy as sa

revision = "20260909_snapshot_mid_price_spread_nullable"
down_revision = "20260909_orderbook_depth_and_market_bounds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("market_snapshots") as batch_op:
        batch_op.alter_column("mid_price", existing_type=sa.Float(), nullable=True)
        batch_op.alter_column("spread", existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("market_snapshots") as batch_op:
        batch_op.alter_column("spread", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("mid_price", existing_type=sa.Float(), nullable=False)
