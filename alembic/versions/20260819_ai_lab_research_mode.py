"""Add explicit AI Lab STANDARD/RESEARCH mode and research run statuses.

Revision ID: 20260819_ai_lab_research_mode
Revises: 20260815_funnel_raw_opinion
"""

from alembic import op
import sqlalchemy as sa


revision = "20260819_ai_lab_research_mode"
down_revision = "20260815_funnel_raw_opinion"
branch_labels = None
depends_on = None


_RUN_STATUS = (
    "('DRAFT', 'QUEUED', 'PLANNING', 'RUNNING', 'EVALUATING', 'PAUSED', "
    "'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'COMPLETED', 'INSUFFICIENT_DATA', "
    "'RESEARCH_PROVISIONAL', 'INSUFFICIENT_EVIDENCE', 'TECHNICAL_INVALID', "
    "'FAILED', 'REJECTED', 'CANCELLED', 'ROLLED_BACK')"
)


def upgrade() -> None:
    op.add_column(
        "ai_optimization_runs",
        sa.Column("mode", sa.String(length=16), nullable=False, server_default="STANDARD"),
    )
    op.execute(
        "ALTER TABLE ai_optimization_runs "
        "DROP CONSTRAINT IF EXISTS ck_ai_runs_status"
    )
    op.create_check_constraint(
        "ck_ai_runs_status",
        "ai_optimization_runs",
        f"status IN {_RUN_STATUS}",
    )
    op.create_check_constraint(
        "ck_ai_runs_mode",
        "ai_optimization_runs",
        "mode IN ('STANDARD', 'RESEARCH')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_ai_runs_mode", "ai_optimization_runs", type_="check")
    op.execute(
        "ALTER TABLE ai_optimization_runs "
        "DROP CONSTRAINT IF EXISTS ck_ai_runs_status"
    )
    op.create_check_constraint(
        "ck_ai_runs_status",
        "ai_optimization_runs",
        "status IN ('DRAFT', 'QUEUED', 'PLANNING', 'RUNNING', 'EVALUATING', "
        "'PAUSED', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'COMPLETED', "
        "'INSUFFICIENT_DATA', 'FAILED', 'REJECTED', 'CANCELLED', 'ROLLED_BACK')",
    )
    op.drop_column("ai_optimization_runs", "mode")
