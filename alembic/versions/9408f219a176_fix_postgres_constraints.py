"""fix_postgres_constraints

Revision ID: 9408f219a176
Revises: b82a7c6bfe1f
Create Date: 2026-07-26 18:13:21.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9408f219a176'
down_revision = 'b82a7c6bfe1f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Update MarketSnapshot constraints
    # Ensure flip_vs_final is nullable
    with op.batch_alter_table('market_snapshots', schema=None) as batch_op:
        batch_op.alter_column('flip_vs_final',
               existing_type=sa.BOOLEAN(),
               nullable=True)
        # Recreate the check constraint for outcomes
        batch_op.drop_constraint('ck_market_snapshot_outcome', type_='check')
        batch_op.create_check_constraint(
            'ck_market_snapshot_outcome',
            "final_outcome IN ('PENDING', 'YES', 'NO', 'INVALID')"
        )

    # 2. Update CryptoCandles constraint for Postgres compatibility
    # The previous migration used `is_closed = false` which fails on Postgres.
    with op.batch_alter_table('crypto_candles', schema=None) as batch_op:
        batch_op.drop_constraint('ck_crypto_candles_close_time', type_='check')
        # We use IS NOT TRUE instead of = false to handle both false and NULL securely
        batch_op.create_check_constraint(
            'ck_crypto_candles_close_time',
            '(is_closed IS NOT TRUE) OR (close_time IS NOT NULL)'
        )


def downgrade() -> None:
    with op.batch_alter_table('crypto_candles', schema=None) as batch_op:
        batch_op.drop_constraint('ck_crypto_candles_close_time', type_='check')
        # Revert to the broken constraint
        batch_op.create_check_constraint(
            'ck_crypto_candles_close_time',
            '(is_closed = false) OR (is_closed IS NULL) OR (close_time IS NOT NULL)'
        )

    with op.batch_alter_table('market_snapshots', schema=None) as batch_op:
        batch_op.drop_constraint('ck_market_snapshot_outcome', type_='check')
        batch_op.create_check_constraint(
            'ck_market_snapshot_outcome',
            "final_outcome IN ('YES', 'NO', 'INVALID')"
        )
        batch_op.alter_column('flip_vs_final',
               existing_type=sa.BOOLEAN(),
               nullable=False)
