"""add indexes to trade_history

Revision ID: c0a5bfd9e919
Revises: 2454976542ef
Create Date: 2026-06-29 08:21:10.113144+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c0a5bfd9e919'
down_revision: Union[str, None] = '2454976542ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS idx_trade_history_created_at ON trade_history (created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_trade_history_status ON trade_history (status)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_trade_history_status")
    op.execute("DROP INDEX IF EXISTS idx_trade_history_created_at")
