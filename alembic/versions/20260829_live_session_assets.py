"""Store selected assets on LIVE trading sessions.

Revision ID: 20260829_live_session_assets
Revises: 20260828_merge_ai_lab_mrf_heads
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260829_live_session_assets"
down_revision = "20260828_merge_ai_lab_mrf_heads"
branch_labels = None
depends_on = None


DEFAULT_ASSETS = '["BTC","ETH","SOL","XRP","DOGE"]'


def upgrade() -> None:
    op.add_column(
        "live_trading_sessions",
        sa.Column(
            "selected_assets",
            sa.JSON().with_variant(postgresql.JSONB, "postgresql"),
            nullable=False,
            server_default=sa.text(f"'{DEFAULT_ASSETS}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("live_trading_sessions", "selected_assets")
