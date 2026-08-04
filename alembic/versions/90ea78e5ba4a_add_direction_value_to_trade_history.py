"""add direction_value to trade history

Revision ID: 90ea78e5ba4a
Revises: e2a4f1b3c9d6
Create Date: 2026-08-04 08:37:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '90ea78e5ba4a'
down_revision = 'e2a4f1b3c9d6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('direction_value', sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'direction_value')
