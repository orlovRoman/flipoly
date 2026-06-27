"""add kelly fields to trade_history

Revision ID: 2454976542ed
Revises: 72fb8ee4d283
Create Date: 2026-06-27 07:07:07.960910+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2454976542ed'
down_revision: Union[str, None] = '72fb8ee4d283'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('kelly_fraction', sa.Float(), nullable=True))
    op.add_column('trade_history', sa.Column('kelly_multiplier', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'kelly_multiplier')
    op.drop_column('trade_history', 'kelly_fraction')
