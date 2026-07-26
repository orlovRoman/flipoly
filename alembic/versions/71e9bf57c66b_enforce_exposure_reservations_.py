"""enforce exposure reservations constraints

Revision ID: 71e9bf57c66b
Revises: e2f3g4h5i6j7
Create Date: 2026-07-26 16:37:11.423640+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '71e9bf57c66b'
down_revision: Union[str, None] = 'e2f3g4h5i6j7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update execution_requests trade_history_id to NOT NULL
    op.alter_column('execution_requests', 'trade_history_id', existing_type=sa.Integer(), nullable=False)
    
    # 2. Fix the check constraint on execution_requests
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute("ALTER TABLE execution_requests DROP CONSTRAINT IF EXISTS ck_execution_request_trade_reference")
    else:
        with op.batch_alter_table('execution_requests') as batch_op:
            try:
                batch_op.drop_constraint('ck_execution_request_trade_reference', type_='check')
            except Exception:
                pass
    op.create_check_constraint(
        'ck_execution_request_trade_reference',
        'execution_requests',
        "intent IN ('OPEN', 'CLOSE') AND trade_history_id IS NOT NULL"
    )

    # 3. Add missing constraints to exposure_reservations
    op.alter_column('exposure_reservations', 'market_id', existing_type=sa.String(128), nullable=False)
    op.alter_column('exposure_reservations', 'expires_at', existing_type=sa.DateTime(timezone=True), nullable=False)
    op.alter_column('exposure_reservations', 'created_at', existing_type=sa.DateTime(timezone=True), nullable=False)
    
    # 4. Drop trade_id if it exists
    # Check if column exists before dropping to make it idempotent
    conn = op.get_bind()
    insp = sa.inspect(conn)
    columns = [col['name'] for col in insp.get_columns('exposure_reservations')]
    if 'trade_id' in columns:
        op.drop_index(op.f('ix_exposure_reservations_trade_id'), table_name='exposure_reservations')
        op.drop_column('exposure_reservations', 'trade_id')

def downgrade() -> None:
    # 1. Restore trade_id to exposure_reservations
    op.add_column('exposure_reservations', sa.Column('trade_id', sa.String(length=128), nullable=True))
    op.create_index(op.f('ix_exposure_reservations_trade_id'), 'exposure_reservations', ['trade_id'], unique=False)

    # 2. Revert NOT NULL on exposure_reservations
    op.alter_column('exposure_reservations', 'created_at', existing_type=sa.DateTime(timezone=True), nullable=True)
    op.alter_column('exposure_reservations', 'expires_at', existing_type=sa.DateTime(timezone=True), nullable=True)
    op.alter_column('exposure_reservations', 'market_id', existing_type=sa.String(128), nullable=True)

    # 3. Revert check constraint
    op.drop_constraint('ck_execution_request_trade_reference', 'execution_requests', type_='check')
    op.create_check_constraint(
        'ck_execution_request_trade_reference',
        'execution_requests',
        "(intent = 'OPEN' AND trade_history_id IS NULL) OR (intent = 'CLOSE' AND trade_history_id IS NOT NULL)"
    )

    # 4. Revert execution_requests trade_history_id NOT NULL
    op.alter_column('execution_requests', 'trade_history_id', existing_type=sa.Integer(), nullable=True)
