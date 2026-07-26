"""execution_requests_iteration2

Revision ID: f3d4e5f6a7b8
Revises: e2c3d4e5f6a7
Create Date: 2026-07-26 20:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f3d4e5f6a7b8'
down_revision = 'e2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old indexes (can be done directly)
    op.drop_index('uq_active_open_request', table_name='execution_requests', postgresql_where="(intent = 'OPEN') AND (state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN'))")
    op.drop_index('uq_active_close_request', table_name='execution_requests', postgresql_where="(intent = 'CLOSE') AND (state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN'))")

    with op.batch_alter_table('execution_requests') as batch_op:
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column('requested_mode', sa.String(length=32), server_default='PAPER', nullable=False))
        batch_op.add_column(sa.Column('trigger_reason', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('requested_shares', sa.Numeric(precision=38, scale=18), nullable=True))

        batch_op.create_unique_constraint('uq_execution_request_idempotency', ['idempotency_key'])
        
        batch_op.create_check_constraint(
            'ck_execution_request_trade_reference',
            "(intent = 'OPEN' AND trade_history_id IS NULL) OR (intent = 'CLOSE' AND trade_history_id IS NOT NULL)"
        )
        batch_op.create_check_constraint(
            'ck_execution_request_mode',
            "requested_mode IN ('PAPER', 'SHADOW', 'LIVE')"
        )
        batch_op.create_check_constraint(
            'ck_execution_request_positive_shares',
            "requested_shares IS NULL OR requested_shares > 0"
        )
        batch_op.create_foreign_key('fk_execution_requests_trade_history_id', 'trade_history', ['trade_history_id'], ['id'], ondelete='RESTRICT')

    # Recreate indexes with new states
    op.create_index('uq_active_open_request', 'execution_requests', ['market_id'], unique=True, postgresql_where=sa.text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"), sqlite_where=sa.text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"))
    op.create_index('uq_active_close_request', 'execution_requests', ['trade_history_id'], unique=True, postgresql_where=sa.text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"), sqlite_where=sa.text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"))


def downgrade() -> None:
    op.drop_index('uq_active_open_request', table_name='execution_requests', postgresql_where=sa.text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"))
    op.drop_index('uq_active_close_request', table_name='execution_requests', postgresql_where=sa.text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')"))

    with op.batch_alter_table('execution_requests') as batch_op:
        batch_op.drop_constraint('fk_execution_requests_trade_history_id', type_='foreignkey')
        batch_op.drop_constraint('ck_execution_request_trade_reference', type_='check')
        batch_op.drop_constraint('ck_execution_request_mode', type_='check')
        batch_op.drop_constraint('ck_execution_request_positive_shares', type_='check')
        batch_op.drop_constraint('uq_execution_request_idempotency', type_='unique')
        
        batch_op.drop_column('requested_shares')
        batch_op.drop_column('trigger_reason')
        batch_op.drop_column('requested_mode')
        batch_op.drop_column('idempotency_key')

    op.create_index('uq_active_open_request', 'execution_requests', ['market_id'], unique=True, postgresql_where=sa.text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN')"), sqlite_where=sa.text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN')"))
    op.create_index('uq_active_close_request', 'execution_requests', ['trade_history_id'], unique=True, postgresql_where=sa.text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN')"), sqlite_where=sa.text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN')"))
