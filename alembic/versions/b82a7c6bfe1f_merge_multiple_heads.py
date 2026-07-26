"""Merge multiple heads

Revision ID: b82a7c6bfe1f
Revises: c4df36df0c84, ffd266c695fc
Create Date: 2026-07-26 10:55:05.552164+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b82a7c6bfe1f'
down_revision: Union[str, None] = ('c4df36df0c84', 'ffd266c695fc')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
