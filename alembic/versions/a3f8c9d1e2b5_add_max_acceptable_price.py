"""add_max_acceptable_price

Revision ID: a3f8c9d1e2b5
Revises: a3f8c9d1e2b4
Create Date: 2026-08-03 13:25:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f8c9d1e2b5'
down_revision: Union[str, None] = 'a3f8c9d1e2b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update decision_funnel_log
    with op.batch_alter_table('decision_funnel_log', schema=None) as batch_op:
        batch_op.add_column(sa.Column('max_acceptable_price', sa.Float(), nullable=True))

    # 2. Update live_mirror_candidates
    with op.batch_alter_table('live_mirror_candidates', schema=None) as batch_op:
        batch_op.add_column(sa.Column('max_acceptable_price', sa.Float(), nullable=True))


def downgrade() -> None:
    # 1. Update live_mirror_candidates
    with op.batch_alter_table('live_mirror_candidates', schema=None) as batch_op:
        batch_op.drop_column('max_acceptable_price')

    # 2. Update decision_funnel_log
    with op.batch_alter_table('decision_funnel_log', schema=None) as batch_op:
        batch_op.drop_column('max_acceptable_price')
