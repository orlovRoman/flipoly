"""add_crypto_candles

Revision ID: f7b5ac2856cb
Revises: f1a2b3c4d5e7
Create Date: 2026-07-04 14:47:40.350443+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7b5ac2856cb'
down_revision: Union[str, None] = 'f1a2b3c4d5e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('crypto_candles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('interval', sa.String(length=8), nullable=False),
        sa.Column('open_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('taker_buy_volume', sa.Float(), nullable=True),
        sa.Column('source', sa.String(length=16), nullable=False, server_default='binance'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'interval', 'open_time', name='uix_crypto_candle')
    )
    op.create_index('idx_crypto_candles_open_time', 'crypto_candles', ['open_time'], unique=False)
    op.create_index('idx_crypto_candles_symbol_interval', 'crypto_candles', ['symbol', 'interval'], unique=False)

def downgrade() -> None:
    op.drop_index('idx_crypto_candles_symbol_interval', table_name='crypto_candles')
    op.drop_index('idx_crypto_candles_open_time', table_name='crypto_candles')
    op.drop_table('crypto_candles')
