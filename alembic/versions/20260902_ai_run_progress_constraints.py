"""Backfill and constrain AI Lab run progress counters."""

from alembic import op
import sqlalchemy as sa


revision = "20260902_ai_run_progress_constraints"
down_revision = "20260901_weighted_policy_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Legacy rows were created before the ORM default was enforced in the
    # database. Backfill them before adding the NOT NULL constraint.
    op.execute(
        sa.text(
            """
            UPDATE ai_optimization_runs
            SET experiments_completed = 0
            WHERE experiments_completed IS NULL
            """
        )
    )
    op.alter_column(
        "ai_optimization_runs",
        "experiments_completed",
        existing_type=sa.Integer(),
        server_default=sa.text("0"),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "ai_optimization_runs",
        "experiments_completed",
        existing_type=sa.Integer(),
        server_default=None,
        nullable=True,
    )
