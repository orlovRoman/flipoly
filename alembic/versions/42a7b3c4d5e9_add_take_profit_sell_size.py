"""add_take_profit_sell_size

Revision ID: 42a7b3c4d5e9
Revises: 42a7b3c4d5e8
Create Date: 2026-07-14 15:15:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '42a7b3c4d5e9'
down_revision: Union[str, None] = '42a7b3c4d5e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the missing column that was defined in models.py but missed in previous migration
    op.add_column('trade_history', sa.Column('take_profit_sell_size', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'take_profit_sell_size')
