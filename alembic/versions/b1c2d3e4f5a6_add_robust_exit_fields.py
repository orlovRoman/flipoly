\"\"\"add robust exit fields

Revision ID: b1c2d3e4f5a6
Revises: a1b2c3d4e5f6
Create Date: 2026-07-26 18:59:00.000000

\"\"\"
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'b1c2d3e4f5a6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Update stop_loss and take_profit sell size types
    op.alter_column('trade_history', 'stop_loss_sell_size',
               existing_type=sa.Float(),
               type_=sa.Numeric(precision=38, scale=18),
               existing_nullable=True)
               
    # Add if missing (because we added take_profit_sell_size manually in production, but local might not have it)
    conn = op.get_bind()
    insp = sa.inspect(conn)
    cols = [c['name'] for c in insp.get_columns('trade_history')]
    if 'take_profit_sell_size' not in cols:
        op.add_column('trade_history', sa.Column('take_profit_sell_size', sa.Numeric(precision=38, scale=18), nullable=True))
    else:
        op.alter_column('trade_history', 'take_profit_sell_size',
                   existing_type=sa.Float(),
                   type_=sa.Numeric(precision=38, scale=18),
                   existing_nullable=True)

    # 2. Add accounting and exit concurrency fields
    op.add_column('trade_history', sa.Column('position_accounting_version', sa.SmallInteger(), server_default='0', nullable=False))
    op.add_column('trade_history', sa.Column('entry_filled_shares', sa.Numeric(precision=38, scale=18), nullable=True))
    op.add_column('trade_history', sa.Column('entry_cost_usdc', sa.Numeric(precision=38, scale=18), nullable=True))
    op.add_column('trade_history', sa.Column('remaining_shares', sa.Numeric(precision=38, scale=18), nullable=True))
    op.add_column('trade_history', sa.Column('realized_pnl_usdc', sa.Numeric(precision=38, scale=18), server_default='0', nullable=False))
    
    op.add_column('trade_history', sa.Column('exit_attempt_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('trade_history', sa.Column('exit_claimed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('trade_history', sa.Column('last_exit_error', sa.Text(), nullable=True))

    # 3. Create chain_transactions table
    op.create_table('chain_transactions',
        sa.Column('tx_hash', sa.String(length=128), nullable=False),
        sa.Column('attempt_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('operation', sa.String(length=32), nullable=False),
        sa.Column('network', sa.String(length=32), nullable=False),
        sa.Column('gas_paid_native', sa.Numeric(precision=38, scale=18), nullable=True),
        sa.Column('gas_paid_usdc', sa.Numeric(precision=38, scale=18), nullable=True),
        sa.Column('paid_by', sa.String(length=16), nullable=False),
        sa.Column('confirmed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('tx_hash')
    )

    # 4. Add constraints
    op.create_check_constraint(
        'ck_trade_position_accounting_initialized',
        'trade_history',
        'position_accounting_version = 0 OR (remaining_shares IS NOT NULL AND realized_pnl_usdc IS NOT NULL)'
    )

    # 5. Fix corrupted candles
    op.execute(
        \"\"\"
        UPDATE crypto_candles
        SET is_closed = false, close_time = NULL
        WHERE is_closed = true AND close_time = open_time;
        \"\"\"
    )

def downgrade() -> None:
    pass
