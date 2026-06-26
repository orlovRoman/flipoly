"""add features to model_registry

Revision ID: 60f588a4ef77
Revises: 48db8ee4d282
Create Date: 2026-06-26 12:45:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '60f588a4ef77'
down_revision: Union[str, None] = '48db8ee4d282'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_registry', sa.Column('features', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_registry', 'features')
