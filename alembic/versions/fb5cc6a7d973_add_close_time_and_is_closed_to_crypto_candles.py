"""add close_time and is_closed to crypto_candles

Revision ID: fb5cc6a7d973
Revises: eb4cc6a7d972
Create Date: 2026-07-26 13:45:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'fb5cc6a7d973'
down_revision = 'eb4cc6a7d972'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('crypto_candles', sa.Column('close_time', sa.DateTime(timezone=True), nullable=True))
    op.add_column('crypto_candles', sa.Column('is_closed', sa.Boolean(), nullable=True))
    
    # Update existing records to is_closed = True (assuming old ones are historical and closed)
    # Leave old candles as NULL until re-fetched from Binance

def downgrade() -> None:
    op.drop_column('crypto_candles', 'is_closed')
    op.drop_column('crypto_candles', 'close_time')

