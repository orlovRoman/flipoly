"""Merge AI run progress and weighted policy migration heads.

Revision ID: 20260905_merge_weighted_policy_heads
Revises: 20260902_ai_run_progress_constraints, 20260904_weighted_min_net_ev
"""

from alembic import op


revision = "20260905_merge_weighted_policy_heads"
down_revision = (
    "20260902_ai_run_progress_constraints",
    "20260904_weighted_min_net_ev",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
