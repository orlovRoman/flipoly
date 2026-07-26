"""update_exposure_reservations

Revision ID: c1d2e3f4g5h6
Revises: b97b8138716e
Create Date: 2026-07-26 21:51:54.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c1d2e3f4g5h6'
down_revision: Union[str, None] = 'b97b8138716e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. add new columns to exposure_reservations
    op.add_column('exposure_reservations', sa.Column('trade_id', sa.String(length=128), nullable=True))
    op.add_column('exposure_reservations', sa.Column('market_id', sa.String(length=128), nullable=True))
    op.add_column('exposure_reservations', sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('exposure_reservations', sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False))

    # 2. copy data if any? or just drop old columns
    op.drop_column('exposure_reservations', 'asset')
    op.drop_column('exposure_reservations', 'reserved_at')
    op.drop_column('exposure_reservations', 'released_at')
    op.drop_column('exposure_reservations', 'request_id')

    # 3. create indices
    op.create_index(op.f('ix_exposure_reservations_market_id'), 'exposure_reservations', ['market_id'], unique=False)
    op.create_index(op.f('ix_exposure_reservations_trade_id'), 'exposure_reservations', ['trade_id'], unique=False)


def downgrade() -> None:
    # 1. drop indices
    op.drop_index(op.f('ix_exposure_reservations_trade_id'), table_name='exposure_reservations')
    op.drop_index(op.f('ix_exposure_reservations_market_id'), table_name='exposure_reservations')

    # 2. add old columns
    op.add_column('exposure_reservations', sa.Column('request_id', postgresql.UUID(), autoincrement=False, nullable=True))
    op.add_column('exposure_reservations', sa.Column('released_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True))
    op.add_column('exposure_reservations', sa.Column('reserved_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True))
    op.add_column('exposure_reservations', sa.Column('asset', sa.VARCHAR(length=32), autoincrement=False, nullable=True))

    # 3. drop new columns
    op.drop_column('exposure_reservations', 'created_at')
    op.drop_column('exposure_reservations', 'expires_at')
    op.drop_column('exposure_reservations', 'market_id')
    op.drop_column('exposure_reservations', 'trade_id')
