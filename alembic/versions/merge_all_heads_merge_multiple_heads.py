"""merge all heads

Revision ID: merge_all_heads
Revises: 'f3d4e5f6a7b8', '003', '90ea78e5ba4a', 'ffd266c695fc', '12845e0151a9', 'e2f3g4h5i6j7'
Create Date: 2026-08-04 08:50:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge_all_heads'
down_revision = ('f3d4e5f6a7b8', '003', '90ea78e5ba4a', 'ffd266c695fc', '12845e0151a9', 'e2f3g4h5i6j7')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
