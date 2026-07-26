"""unified_position_status

Revision ID: c4df36df0c84
Revises: fb5cc6a7d973
Create Date: 2026-07-26 07:36:16.658869+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4df36df0c84'
down_revision: Union[str, None] = 'fb5cc6a7d973'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('position_status', sa.String(length=32), server_default='OPEN', nullable=False))
    op.add_column('trade_history', sa.Column('exit_reason', sa.String(length=32), nullable=True))
    op.add_column('trade_history', sa.Column('exit_order_id', sa.String(length=128), nullable=True))
    op.add_column('trade_history', sa.Column('exit_attempts', sa.Integer(), server_default='0', nullable=False))
    op.add_column('trade_history', sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('trade_history', sa.Column('close_price', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'close_price')
    op.drop_column('trade_history', 'closed_at')
    op.drop_column('trade_history', 'exit_attempts')
    op.drop_column('trade_history', 'exit_order_id')
    op.drop_column('trade_history', 'exit_reason')
    op.drop_column('trade_history', 'position_status')
