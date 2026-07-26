"""enforce exposure_reservations

Revision ID: e2f3g4h5i6j7
Revises: d1e2f3g4h5i6
Create Date: 2026-07-26 22:42:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'e2f3g4h5i6j7'
down_revision = 'd1e2f3g4h5i6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    conn = op.get_bind()
    
    # Assert that there are no unrecoverable rows before adding NOT NULL
    unrecoverable = conn.scalar(sa.text("""
        SELECT count(*)
        FROM exposure_reservations
        WHERE request_id IS NULL OR trade_history_id IS NULL
    """))
    if unrecoverable:
        raise RuntimeError(f"Cannot apply constraints: {unrecoverable} exposure_reservations are missing request_id or trade_history_id")

    op.alter_column('exposure_reservations', 'request_id',
               existing_type=postgresql.UUID(as_uuid=True),
               nullable=False)
    op.alter_column('exposure_reservations', 'trade_history_id',
               existing_type=sa.INTEGER(),
               nullable=False)

    op.create_unique_constraint('uq_exposure_reservations_request_id', 'exposure_reservations', ['request_id'])

def downgrade() -> None:
    op.drop_constraint('uq_exposure_reservations_request_id', 'exposure_reservations', type_='unique')
    
    op.alter_column('exposure_reservations', 'trade_history_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.alter_column('exposure_reservations', 'request_id',
               existing_type=postgresql.UUID(as_uuid=True),
               nullable=True)
