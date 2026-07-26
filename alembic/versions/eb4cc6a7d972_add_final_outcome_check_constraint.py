"""add final_outcome check constraint

Revision ID: eb4cc6a7d972
Revises: ea2cc6a7d971
Create Date: 2026-07-26 13:42:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'eb4cc6a7d972'
down_revision = 'ea2cc6a7d971'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Adding CheckConstraint to market_snapshots.final_outcome
    op.create_check_constraint(
        'ck_market_snapshot_outcome',
        'market_snapshots',
        "final_outcome IN ('PENDING', 'YES', 'NO', 'INVALID')"
    )

def downgrade() -> None:
    op.drop_constraint('ck_market_snapshot_outcome', 'market_snapshots', type_='check')
