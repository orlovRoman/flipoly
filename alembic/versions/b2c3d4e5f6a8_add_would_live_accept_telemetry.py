"""add would_live_accept, p_flip_raw, entry_model_ece to trade_history and decision_funnel_log

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f7
Create Date: 2026-08-09 14:35:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6a8'
down_revision = 'a1b2c3d4e5f7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. TradeHistory
    op.add_column('trade_history', sa.Column('would_live_accept', sa.Boolean(), nullable=True))
    op.add_column('trade_history', sa.Column('p_flip_raw', sa.Float(), nullable=True))
    op.add_column('trade_history', sa.Column('entry_model_ece', sa.Float(), nullable=True))

    # 2. DecisionFunnelLog
    op.add_column('decision_funnel_log', sa.Column('would_live_accept', sa.Boolean(), nullable=True))
    op.add_column('decision_funnel_log', sa.Column('p_flip_raw', sa.Float(), nullable=True))
    op.add_column('decision_funnel_log', sa.Column('entry_model_ece', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('decision_funnel_log', 'entry_model_ece')
    op.drop_column('decision_funnel_log', 'p_flip_raw')
    op.drop_column('decision_funnel_log', 'would_live_accept')

    op.drop_column('trade_history', 'entry_model_ece')
    op.drop_column('trade_history', 'p_flip_raw')
    op.drop_column('trade_history', 'would_live_accept')
