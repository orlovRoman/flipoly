"""enforce_closed_candle_constraints

Revision ID: ffd266c695fc
Revises: fb5cc6a7d973
Create Date: 2026-07-26 16:15:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ffd266c695fc'
down_revision = 'fb5cc6a7d973'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add check constraint: a closed candle MUST have a close_time.
    # is_closed = false OR is_closed IS NULL OR close_time IS NOT NULL
    with op.batch_alter_table('crypto_candles') as batch_op:
        batch_op.create_check_constraint(
            'ck_crypto_candles_close_time',
            '(is_closed = 0) OR (is_closed IS NULL) OR (close_time IS NOT NULL)'
        )


def downgrade() -> None:
    with op.batch_alter_table('crypto_candles') as batch_op:
        batch_op.drop_constraint('ck_crypto_candles_close_time', type_='check')

