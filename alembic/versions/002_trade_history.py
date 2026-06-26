"""TradeHistory table

Revision ID: 002
Revises: 001
Create Date: 2026-06-25 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'trade_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('market_id', sa.String(length=128), nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('outcome_bought', sa.String(length=16), nullable=False),
        sa.Column('amount_usdc', sa.Float(), nullable=False),
        sa.Column('executed_price', sa.Float(), nullable=False),
        sa.Column('predicted_flip_prob', sa.Float(), nullable=False),
        sa.Column('active_features', sa.String(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('error_msg', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_trade_history_market_id', 'trade_history', ['market_id'], unique=False)
    op.create_index('idx_trade_history_created_at', 'trade_history', ['created_at'], unique=False)

def downgrade() -> None:
    pass
