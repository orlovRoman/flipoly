"""add outsider strategy columns

Revision ID: 93bf82b7c9b7
Revises: f3d4e5f6a7b8
Create Date: 2026-07-26 13:53:23.425687+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '93bf82b7c9b7'
down_revision: Union[str, None] = 'f3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('strategy_type', sa.String(length=32), nullable=True))
    op.add_column('trade_history', sa.Column('market_role', sa.String(length=16), nullable=True))
    op.add_column('trade_history', sa.Column('p_flip_effective', sa.Float(), nullable=True))
    op.add_column('trade_history', sa.Column('p_win_effective', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'p_win_effective')
    op.drop_column('trade_history', 'p_flip_effective')
    op.drop_column('trade_history', 'market_role')
    op.drop_column('trade_history', 'strategy_type')
