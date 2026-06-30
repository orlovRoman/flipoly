"""add bid and ask to market snapshot

Revision ID: d0be6d68b1b5
Revises: d0be6d68b1b4
Create Date: 2026-06-30 11:15:00.000000+00:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'd0be6d68b1b5'
down_revision: Union[str, None] = 'd0be6d68b1b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('market_snapshots', sa.Column('best_bid', sa.Float(), nullable=True))
    op.add_column('market_snapshots', sa.Column('best_ask', sa.Float(), nullable=True))

def downgrade() -> None:
    op.drop_column('market_snapshots', 'best_ask')
    op.drop_column('market_snapshots', 'best_bid')
