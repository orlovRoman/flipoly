"""add_interval_to_model_registry

Revision ID: 5b1cefabc64a
Revises: 3c0cefabc64a
Create Date: 2026-07-04 16:59:48.001052+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b1cefabc64a'
down_revision: Union[str, None] = '3c0cefabc64a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_registry', sa.Column('interval', sa.String(length=5), nullable=False, server_default='15m'))


def downgrade() -> None:
    op.drop_column('model_registry', 'interval')
