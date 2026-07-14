"""add_take_profit_columns

Revision ID: 42a7b3c4d5e8
Revises: b34a6b9256f3
Create Date: 2026-07-14 15:10:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '42a7b3c4d5e8'
down_revision: Union[str, None] = 'b34a6b9256f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('take_profit_enabled', sa.Boolean(), nullable=True))
    op.add_column('trade_history', sa.Column('take_profit_multiplier', sa.Float(), nullable=True))
    op.add_column('trade_history', sa.Column('take_profit_price', sa.Float(), nullable=True))
    op.add_column('trade_history', sa.Column('take_profit_status', sa.String(length=20), nullable=True))
    op.add_column('trade_history', sa.Column('take_profit_hit_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('trade_history', sa.Column('take_profit_sell_price', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'take_profit_sell_price')
    op.drop_column('trade_history', 'take_profit_hit_at')
    op.drop_column('trade_history', 'take_profit_status')
    op.drop_column('trade_history', 'take_profit_price')
    op.drop_column('trade_history', 'take_profit_multiplier')
    op.drop_column('trade_history', 'take_profit_enabled')
