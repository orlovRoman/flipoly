"""add_stop_loss_sell_size

Revision ID: a1b2c3d4e5f6
Revises: 9408f219a176
Create Date: 2026-07-26 18:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '9408f219a176'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('trade_history', sa.Column('stop_loss_sell_size', sa.Float(), nullable=True))

def downgrade() -> None:
    op.drop_column('trade_history', 'stop_loss_sell_size')
