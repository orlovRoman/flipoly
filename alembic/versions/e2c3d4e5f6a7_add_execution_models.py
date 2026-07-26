"""add execution models and update constraints

Revision ID: e2c3d4e5f6a7
Revises: b1c2d3e4f5a6
Create Date: 2026-07-26 20:15:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'e2c3d4e5f6a7'
down_revision = 'b1c2d3e4f5a6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Update realized_pnl_usdc to be nullable=True and remove default
    op.alter_column('trade_history', 'realized_pnl_usdc',
               existing_type=sa.Numeric(precision=38, scale=18),
               server_default=None,
               nullable=True)

    # 2. Update the check constraint ck_trade_position_accounting_initialized
    op.drop_constraint('ck_trade_position_accounting_initialized', 'trade_history', type_='check')
    op.create_check_constraint(
        'ck_trade_position_accounting_initialized',
        'trade_history',
        'position_accounting_version = 0 OR (entry_filled_shares IS NOT NULL AND entry_cost_usdc IS NOT NULL AND remaining_shares IS NOT NULL AND realized_pnl_usdc IS NOT NULL)'
    )

    # 3. Create execution_requests
    op.create_table('execution_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('trade_history_id', sa.Integer(), nullable=True),
        sa.Column('intent', sa.String(length=32), nullable=False),
        sa.Column('market_id', sa.String(length=128), nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('outcome_to_buy', sa.String(length=16), nullable=False),
        sa.Column('target_amount_usdc', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column('max_slippage_pct', sa.Float(), nullable=False),
        sa.Column('ttl_seconds', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('error_reason', sa.Text(), nullable=True),
        sa.Column('filled_shares', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column('filled_cost_usdc', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('uq_active_open_request', 'execution_requests', ['market_id'], unique=True, postgresql_where=sa.text("intent = 'OPEN' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN')"))
    op.create_index('uq_active_close_request', 'execution_requests', ['trade_history_id'], unique=True, postgresql_where=sa.text("intent = 'CLOSE' AND state IN ('READY', 'CLAIMED', 'SUBMITTING', 'UNKNOWN')"))

    # 4. Create execution_attempts
    op.create_table('execution_attempts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('gateway', sa.String(length=32), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('tx_hash', sa.String(length=128), nullable=True),
        sa.Column('error_msg', sa.Text(), nullable=True),
        sa.Column('raw_response', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['request_id'], ['execution_requests.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # 5. Create execution_fills
    op.create_table('execution_fills',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('attempt_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('price', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column('shares', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column('fee_usdc', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['attempt_id'], ['execution_attempts.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # 6. Create execution_approvals
    op.create_table('execution_approvals',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('approval_hash', sa.String(length=128), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('consumed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('consumed_by_ip', sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(['request_id'], ['execution_requests.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('approval_hash')
    )

    # 7. Create execution_events
    op.create_table('execution_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['request_id'], ['execution_requests.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # 8. Create exposure_reservations
    op.create_table('exposure_reservations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('asset', sa.String(length=32), nullable=False),
        sa.Column('amount_usdc', sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column('reserved_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('released_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['request_id'], ['execution_requests.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 9. Update chain_transactions foreign key
    op.create_foreign_key('fk_chain_transactions_attempt_id', 'chain_transactions', 'execution_attempts', ['attempt_id'], ['id'])

def downgrade() -> None:
    op.drop_constraint('fk_chain_transactions_attempt_id', 'chain_transactions', type_='foreignkey')
    op.drop_table('exposure_reservations')
    op.drop_table('execution_events')
    op.drop_table('execution_approvals')
    op.drop_table('execution_fills')
    op.drop_table('execution_attempts')
    op.drop_index('uq_active_close_request', table_name='execution_requests')
    op.drop_index('uq_active_open_request', table_name='execution_requests')
    op.drop_table('execution_requests')
    
    op.drop_constraint('ck_trade_position_accounting_initialized', 'trade_history', type_='check')
    op.create_check_constraint(
        'ck_trade_position_accounting_initialized',
        'trade_history',
        'position_accounting_version = 0 OR (remaining_shares IS NOT NULL AND realized_pnl_usdc IS NOT NULL)'
    )
    
    op.alter_column('trade_history', 'realized_pnl_usdc',
               existing_type=sa.Numeric(precision=38, scale=18),
               server_default='0',
               nullable=False)
