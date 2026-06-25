"""initial schema

Revision ID: 001
Revises: 
Create Date: 2026-06-25 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # market_snapshots
    op.create_table(
        'market_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('market_id', sa.String(length=128), nullable=False),
        sa.Column('time_left_min', sa.Float(), nullable=False),
        sa.Column('mid_price', sa.Float(), nullable=False),
        sa.Column('spread', sa.Float(), nullable=False),
        sa.Column('volume_5min', sa.Float(), nullable=False),
        sa.Column('price_velocity', sa.Float(), nullable=False),
        sa.Column('hour_of_day', sa.Integer(), nullable=False),
        sa.Column('final_outcome', sa.String(length=16), nullable=False),
        sa.Column('flip_vs_final', sa.Boolean(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_market_snapshots_asset', 'market_snapshots', ['asset'], unique=False)
    op.create_index('idx_market_snapshots_asset_time', 'market_snapshots', ['asset', 'time_left_min'], unique=False)
    op.create_index('idx_market_snapshots_recorded_at', 'market_snapshots', ['recorded_at'], unique=False)

    # model_registry
    op.create_table(
        'model_registry',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('model_blob', sa.LargeBinary(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('trained_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_model_registry_asset_active', 'model_registry', ['asset', 'is_active'], unique=False)

    # collector_status
    op.create_table(
        'collector_status',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('markets_found', sa.Integer(), nullable=False),
        sa.Column('markets_saved', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('duration_sec', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # runtime_settings
    op.create_table(
        'runtime_settings',
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('value', sa.String(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_by', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('key')
    )

    # live_markets
    op.create_table(
        'live_markets',
        sa.Column('market_id', sa.String(length=128), nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('question', sa.String(), nullable=False),
        sa.Column('end_time_est', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_yes_price', sa.Float(), nullable=False),
        sa.Column('current_no_price', sa.Float(), nullable=False),
        sa.Column('current_spread', sa.Float(), nullable=False),
        sa.Column('volume_5min', sa.Float(), nullable=False),
        sa.Column('price_velocity', sa.Float(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('market_id')
    )
    op.create_index('idx_live_markets_asset', 'live_markets', ['asset'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_live_markets_asset', table_name='live_markets')
    op.drop_table('live_markets')
    op.drop_table('runtime_settings')
    op.drop_table('collector_status')
    op.drop_index('idx_model_registry_asset_active', table_name='model_registry')
    op.drop_table('model_registry')
    op.drop_index('idx_market_snapshots_recorded_at', table_name='market_snapshots')
    op.drop_index('idx_market_snapshots_asset_time', table_name='market_snapshots')
    op.drop_index('idx_market_snapshots_asset', table_name='market_snapshots')
    op.drop_table('market_snapshots')
