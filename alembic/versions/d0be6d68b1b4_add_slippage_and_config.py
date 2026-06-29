"""add slippage_log and strategy_config tables

Revision ID: d0be6d68b1b4
Revises: c0a5bfd9e919
Create Date: 2026-06-29 23:15:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0be6d68b1b4'
down_revision: Union[str, None] = 'c0a5bfd9e919'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create slippage_log table
    op.create_table(
        'slippage_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('trade_id', sa.Integer(), nullable=False),
        sa.Column('market_id', sa.String(length=128), nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('outcome_bought', sa.String(length=16), nullable=False),
        sa.Column('expected_price', sa.Float(), nullable=False),
        sa.Column('executed_price', sa.Float(), nullable=False),
        sa.Column('slippage', sa.Float(), nullable=False),
        sa.Column('slippage_pct', sa.Float(), nullable=False),
        sa.Column('bet_size_usdc', sa.Float(), nullable=False),
        sa.Column('slippage_cost_usdc', sa.Float(), nullable=False),
        sa.Column('mode', sa.String(length=16), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_slippage_log_asset', 'slippage_log', ['asset'], unique=False)
    op.create_index('idx_slippage_log_created_at', 'slippage_log', ['created_at'], unique=False)
    op.create_index('idx_slippage_log_trade_id', 'slippage_log', ['trade_id'], unique=False)

    # 2. Create strategy_config table
    op.create_table(
        'strategy_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('old_value', sa.String(), nullable=True),
        sa.Column('new_value', sa.String(), nullable=False),
        sa.Column('changed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('changed_by', sa.String(length=64), nullable=False),
        sa.Column('source_ip', sa.String(length=64), nullable=True),
        sa.Column('note', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_strategy_config_key', 'strategy_config', ['key'], unique=False)
    op.create_index('idx_strategy_config_changed_at', 'strategy_config', ['changed_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_strategy_config_changed_at', table_name='strategy_config')
    op.drop_index('idx_strategy_config_key', table_name='strategy_config')
    op.drop_table('strategy_config')
    op.drop_index('idx_slippage_log_trade_id', table_name='slippage_log')
    op.drop_index('idx_slippage_log_created_at', table_name='slippage_log')
    op.drop_index('idx_slippage_log_asset', table_name='slippage_log')
    op.drop_table('slippage_log')
