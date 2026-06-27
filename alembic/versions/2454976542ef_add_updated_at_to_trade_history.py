"""add updated_at to trade_history

Revision ID: 2454976542ef
Revises: 2454976542ed
Create Date: 2026-06-27 08:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2454976542ef'
down_revision: Union[str, None] = '2454976542ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_history', 'updated_at')
