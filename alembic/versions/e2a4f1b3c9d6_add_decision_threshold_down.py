"""add decision_threshold_down

Revision ID: e2a4f1b3c9d6
Revises: e2a4f1b3c9d5
Create Date: 2026-08-03 20:17:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e2a4f1b3c9d6'
down_revision = 'e2a4f1b3c9d5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('model_registry', sa.Column('decision_threshold_down', sa.Float(), nullable=True))
    op.add_column('decision_funnel_log', sa.Column('direction_threshold_up', sa.Float(), nullable=True))
    op.add_column('decision_funnel_log', sa.Column('direction_threshold_down', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('decision_funnel_log', 'direction_threshold_down')
    op.drop_column('decision_funnel_log', 'direction_threshold_up')
    op.drop_column('model_registry', 'decision_threshold_down')
