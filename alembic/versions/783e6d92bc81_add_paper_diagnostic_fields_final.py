"""add_paper_diagnostic_fields_final

Revision ID: 783e6d92bc81
Revises: a3f8c9d1e2b5
Create Date: 2026-08-03 09:08:29.725867+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '783e6d92bc81'
down_revision: Union[str, None] = 'a3f8c9d1e2b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('decision_funnel_log', sa.Column('required_direction_model_key', sa.String(length=64), nullable=True))
    op.add_column('decision_funnel_log', sa.Column('direction_p_up', sa.Float(), nullable=True))
    op.add_column('decision_funnel_log', sa.Column('direction_p_down', sa.Float(), nullable=True))

def downgrade() -> None:
    op.drop_column('decision_funnel_log', 'direction_p_down')
    op.drop_column('decision_funnel_log', 'direction_p_up')
    op.drop_column('decision_funnel_log', 'required_direction_model_key')
