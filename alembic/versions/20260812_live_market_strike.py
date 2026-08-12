"""store canonical Polymarket opening strike on live markets"""
from alembic import op
import sqlalchemy as sa

revision = "20260812_live_market_strike"
down_revision = "20260812_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("live_markets", sa.Column("underlying_price", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("live_markets", "underlying_price")
