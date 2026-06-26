"""add baseline to model_registry

Revision ID: 72fb8ee4d283
Revises: 60f588a4ef77
Create Date: 2026-06-26 13:30:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '72fb8ee4d283'
down_revision: Union[str, None] = '60f588a4ef77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_registry', sa.Column('baseline', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_registry', 'baseline')
