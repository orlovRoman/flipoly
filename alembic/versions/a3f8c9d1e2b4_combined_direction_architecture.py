"""combined_direction_architecture

Revision ID: a3f8c9d1e2b4
Revises: 12845e0151a9
Create Date: 2026-08-03 12:55:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f8c9d1e2b4'
down_revision: Union[str, None] = '12845e0151a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update decision_funnel_log
    with op.batch_alter_table('decision_funnel_log', schema=None) as batch_op:
        batch_op.add_column(sa.Column('decision_run_id', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('direction_model_key', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('direction_model_version', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('direction_regime', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('direction_status', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('direction_probability', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('direction_value', sa.String(length=16), nullable=True))
        
        batch_op.add_column(sa.Column('entry_requested_key', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('entry_model_key', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('entry_model_version', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('entry_model_phase', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('entry_model_source', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('entry_status', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('fallback_reason', sa.String(length=128), nullable=True))
        
        batch_op.add_column(sa.Column('p_candidate_win', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('candidate_side', sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column('candidate_ask', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('gross_edge', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cost_buffer', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('net_edge', sa.Float(), nullable=True))
        
        batch_op.add_column(sa.Column('strike_source', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('strike_proxy', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('underlying_price', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('distance_to_strike_pct', sa.Float(), nullable=True))
        
        batch_op.create_index('idx_funnel_direction_model', ['direction_model_key', 'direction_model_version', 'created_at'], unique=False)
        batch_op.create_index('idx_funnel_entry_model', ['entry_model_key', 'entry_model_version', 'created_at'], unique=False)
        batch_op.create_index('idx_funnel_decision_run', ['decision_run_id'], unique=False)

    # 2. Update trade_history
    with op.batch_alter_table('trade_history', schema=None) as batch_op:
        batch_op.add_column(sa.Column('direction_model_key', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('direction_model_version', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('entry_model_key', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('entry_model_version', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('entry_model_source', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('p_candidate_win', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('gross_edge', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cost_buffer', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('net_edge', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('decision_run_id', sa.String(length=64), nullable=True))

    # 3. Update live_mirror_candidates
    with op.batch_alter_table('live_mirror_candidates', schema=None) as batch_op:
        batch_op.add_column(sa.Column('p_candidate_win', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('decision_ask', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('decision_net_edge', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cost_buffer', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('entry_model_source', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('direction_model_key', sa.String(length=64), nullable=True))

    # 4. Update execution_requests
    with op.batch_alter_table('execution_requests', schema=None) as batch_op:
        batch_op.add_column(sa.Column('max_acceptable_price', sa.Numeric(precision=38, scale=18), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('execution_requests', schema=None) as batch_op:
        batch_op.drop_column('max_acceptable_price')

    with op.batch_alter_table('live_mirror_candidates', schema=None) as batch_op:
        batch_op.drop_column('direction_model_key')
        batch_op.drop_column('entry_model_source')
        batch_op.drop_column('cost_buffer')
        batch_op.drop_column('decision_net_edge')
        batch_op.drop_column('decision_ask')
        batch_op.drop_column('p_candidate_win')

    with op.batch_alter_table('trade_history', schema=None) as batch_op:
        batch_op.drop_column('decision_run_id')
        batch_op.drop_column('net_edge')
        batch_op.drop_column('cost_buffer')
        batch_op.drop_column('gross_edge')
        batch_op.drop_column('p_candidate_win')
        batch_op.drop_column('entry_model_source')
        batch_op.drop_column('entry_model_version')
        batch_op.drop_column('entry_model_key')
        batch_op.drop_column('direction_model_version')
        batch_op.drop_column('direction_model_key')

    with op.batch_alter_table('decision_funnel_log', schema=None) as batch_op:
        batch_op.drop_index('idx_funnel_decision_run')
        batch_op.drop_index('idx_funnel_entry_model')
        batch_op.drop_index('idx_funnel_direction_model')
        batch_op.drop_column('distance_to_strike_pct')
        batch_op.drop_column('underlying_price')
        batch_op.drop_column('strike_proxy')
        batch_op.drop_column('strike_source')
        batch_op.drop_column('net_edge')
        batch_op.drop_column('cost_buffer')
        batch_op.drop_column('gross_edge')
        batch_op.drop_column('candidate_ask')
        batch_op.drop_column('candidate_side')
        batch_op.drop_column('p_candidate_win')
        batch_op.drop_column('fallback_reason')
        batch_op.drop_column('entry_status')
        batch_op.drop_column('entry_model_source')
        batch_op.drop_column('entry_model_phase')
        batch_op.drop_column('entry_model_version')
        batch_op.drop_column('entry_model_key')
        batch_op.drop_column('entry_requested_key')
        batch_op.drop_column('direction_value')
        batch_op.drop_column('direction_probability')
        batch_op.drop_column('direction_status')
        batch_op.drop_column('direction_regime')
        batch_op.drop_column('direction_model_version')
        batch_op.drop_column('direction_model_key')
        batch_op.drop_column('decision_run_id')
