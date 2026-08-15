"""Add durable queue/pause/completed states for autonomous AI runs.

Revision ID: 20260815_ai_agent_run_states
Revises: 20260815_ai_config_overlays
"""

from alembic import op

revision = "20260815_ai_agent_run_states"
down_revision = "20260815_ai_config_overlays"
branch_labels = None
depends_on = None

_RUN_STATUS = (
    "('DRAFT', 'QUEUED', 'PLANNING', 'RUNNING', 'EVALUATING', "
    "'PAUSED', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'COMPLETED', "
    "'INSUFFICIENT_DATA', 'FAILED', 'REJECTED', 'CANCELLED', 'ROLLED_BACK')"
)
_OLD_RUN_STATUS = (
    "('DRAFT', 'PLANNING', 'RUNNING', 'EVALUATING', 'SHADOW', "
    "'PENDING_APPROVAL', 'ACTIVE', 'INSUFFICIENT_DATA', 'FAILED', "
    "'REJECTED', 'CANCELLED', 'ROLLED_BACK')"
)


def upgrade() -> None:
    op.execute(
        "ALTER TABLE ai_optimization_runs "
        "DROP CONSTRAINT IF EXISTS ck_ai_runs_status"
    )
    op.create_check_constraint(
        "ck_ai_runs_status",
        "ai_optimization_runs",
        f"status IN {_RUN_STATUS}",
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE ai_optimization_runs "
        "DROP CONSTRAINT IF EXISTS ck_ai_runs_status"
    )
    op.create_check_constraint(
        "ck_ai_runs_status",
        "ai_optimization_runs",
        f"status IN {_OLD_RUN_STATUS}",
    )
