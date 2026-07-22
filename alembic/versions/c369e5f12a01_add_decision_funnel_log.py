"""add_decision_funnel_log

Revision ID: c369e5f12a01
Revises: b258f4d83d08
Create Date: 2026-07-22 23:59:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c369e5f12a01'
down_revision: Union[str, None] = 'b258f4d83d08'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'decision_funnel_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('market_id', sa.String(length=128), nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('trading_mode', sa.String(length=32), nullable=False),
        sa.Column('used_model', sa.String(length=64), nullable=True),
        sa.Column('p_flip', sa.Float(), nullable=True),
        sa.Column('edge', sa.Float(), nullable=True),
        sa.Column('fresh_price', sa.Float(), nullable=True),
        sa.Column('threshold_lower', sa.Float(), nullable=True),
        sa.Column('threshold_upper', sa.Float(), nullable=True),
        sa.Column('min_edge_used', sa.Float(), nullable=True),
        sa.Column('g1_model_loaded', sa.Boolean(), nullable=True),
        sa.Column('g2_price_fetched', sa.Boolean(), nullable=True),
        sa.Column('g3_dead_zone', sa.Boolean(), nullable=True),
        sa.Column('g4_no_flip', sa.Boolean(), nullable=True),
        sa.Column('g5_min_edge', sa.Boolean(), nullable=True),
        sa.Column('g6_price_range', sa.Boolean(), nullable=True),
        sa.Column('g7_crypto_confirm', sa.Boolean(), nullable=True),
        sa.Column('g8_combined_vote', sa.Boolean(), nullable=True),
        sa.Column('final_action', sa.String(length=16), nullable=False),
        sa.Column('skip_reason', sa.String(length=256), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_funnel_asset_created', 'decision_funnel_log', ['asset', 'created_at'], unique=False)
    op.create_index('idx_funnel_market_id', 'decision_funnel_log', ['market_id'], unique=False)
    op.create_index('idx_funnel_trading_mode', 'decision_funnel_log', ['trading_mode', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_funnel_trading_mode', table_name='decision_funnel_log')
    op.drop_index('idx_funnel_market_id', table_name='decision_funnel_log')
    op.drop_index('idx_funnel_asset_created', table_name='decision_funnel_log')
    op.drop_table('decision_funnel_log')
