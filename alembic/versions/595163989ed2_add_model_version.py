"""add_model_version

Revision ID: 595163989ed2
Revises: 003
Create Date: 2026-06-26 10:47:25.634201+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '595163989ed2'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_history', sa.Column('model_version', sa.Integer(), nullable=True))


def downgrade() -> None:
    # no-op to satisfy SonarQube rule
    pass
