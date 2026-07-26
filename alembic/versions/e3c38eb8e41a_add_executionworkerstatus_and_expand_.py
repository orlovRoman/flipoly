"""Add ExecutionWorkerStatus and expand ExecutionEvent

Revision ID: e3c38eb8e41a
Revises: 71e9bf57c66b
Create Date: 2026-07-26 17:49:43.354042+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e3c38eb8e41a'
down_revision: Union[str, None] = '71e9bf57c66b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create execution_worker_status table
    op.create_table(
        'execution_worker_status',
        sa.Column('worker_id', sa.String(length=100), nullable=False),
        sa.Column('execution_mode', sa.String(length=16), nullable=False),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('gateway_ready', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('credentials_loaded', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('wallet_address', sa.String(length=64), nullable=True),
        sa.Column('balance_usdc', sa.Numeric(precision=38, scale=18), nullable=True),
        sa.Column('collateral_allowance_ready', sa.Boolean(), nullable=True),
        sa.Column('conditional_allowance_ready', sa.Boolean(), nullable=True),
        sa.Column('last_error_code', sa.String(length=64), nullable=True),
        sa.Column('last_error_message', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('worker_id')
    )

    # 2. Expand execution_events table
    op.add_column('execution_events', sa.Column('level', sa.String(length=16), server_default='INFO', nullable=False))
    op.add_column('execution_events', sa.Column('message', sa.Text(), server_default='Migrated event', nullable=False))
    op.add_column('execution_events', sa.Column('source', sa.String(length=32), server_default='system', nullable=False))
    op.add_column('execution_events', sa.Column('trade_history_id', sa.Integer(), nullable=True))
    op.add_column('execution_events', sa.Column('attempt_id', sa.UUID(), nullable=True))

    # Alter request_id to be nullable
    op.alter_column('execution_events', 'request_id', existing_type=sa.UUID(), nullable=True)
    
    # Drop old foreign key for request_id (assuming default naming convention, which is often tricky)
    # Usually alembic autogenerate creates FKs without name if not specified.
    # In postgres, we can use a raw SQL to drop the FK if we know its name, but dropping FKs in alembic
    # requires knowing the constraint name. Let's try dropping the constraint by naming convention.
    op.drop_constraint('execution_events_request_id_fkey', 'execution_events', type_='foreignkey')

    # Recreate foreign keys with ON DELETE SET NULL
    op.create_foreign_key(
        'fk_execution_events_request_id', 'execution_events', 'execution_requests', ['request_id'], ['id'], ondelete='SET NULL'
    )
    op.create_foreign_key(
        'fk_execution_events_trade_history_id', 'execution_events', 'trade_history', ['trade_history_id'], ['id'], ondelete='SET NULL'
    )
    op.create_foreign_key(
        'fk_execution_events_attempt_id', 'execution_events', 'execution_attempts', ['attempt_id'], ['id'], ondelete='SET NULL'
    )

    # Constraints and Indexes
    op.create_check_constraint(
        'ck_execution_event_level', 'execution_events',
        "level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')"
    )
    op.create_index('ix_execution_events_created_at', 'execution_events', ['created_at'], unique=False)
    op.create_index('ix_execution_events_request_time', 'execution_events', ['request_id', 'created_at'], unique=False)
    op.create_index('ix_execution_events_trade_time', 'execution_events', ['trade_history_id', 'created_at'], unique=False)
    op.create_index('ix_execution_events_type_time', 'execution_events', ['event_type', 'created_at'], unique=False)

def downgrade() -> None:
    op.drop_index('ix_execution_events_type_time', table_name='execution_events')
    op.drop_index('ix_execution_events_trade_time', table_name='execution_events')
    op.drop_index('ix_execution_events_request_time', table_name='execution_events')
    op.drop_index('ix_execution_events_created_at', table_name='execution_events')
    op.drop_constraint('ck_execution_event_level', 'execution_events', type_='check')

    op.drop_constraint('fk_execution_events_attempt_id', 'execution_events', type_='foreignkey')
    op.drop_constraint('fk_execution_events_trade_history_id', 'execution_events', type_='foreignkey')
    op.drop_constraint('fk_execution_events_request_id', 'execution_events', type_='foreignkey')

    op.create_foreign_key('execution_events_request_id_fkey', 'execution_events', 'execution_requests', ['request_id'], ['id'])
    op.alter_column('execution_events', 'request_id', existing_type=sa.UUID(), nullable=False)

    op.drop_column('execution_events', 'attempt_id')
    op.drop_column('execution_events', 'trade_history_id')
    op.drop_column('execution_events', 'source')
    op.drop_column('execution_events', 'message')
    op.drop_column('execution_events', 'level')

    op.drop_table('execution_worker_status')
