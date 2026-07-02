"""add ece to model registry

Revision ID: e1cf7d68b1b6
Revises: d0be6d68b1b5
Create Date: 2026-07-02 12:15:00.000000+00:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'e1cf7d68b1b6'
down_revision: Union[str, None] = 'd0be6d68b1b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('model_registry', sa.Column('ece', sa.Float(), nullable=True))

def downgrade() -> None:
    op.drop_column('model_registry', 'ece')
