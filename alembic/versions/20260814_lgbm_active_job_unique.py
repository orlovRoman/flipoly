"""enforce one active LightGBM training job per symbol

Revision ID: 20260814_lgbm_active_job_unique
Revises: 20260813_ai_step_audit
"""

from alembic import op
import sqlalchemy as sa


revision = "20260814_lgbm_active_job_unique"
down_revision = "20260813_ai_step_audit"
branch_labels = None
depends_on = None


_ACTIVE_STATUS = "status IN ('QUEUED', 'RUNNING')"


def upgrade() -> None:
    op.create_index(
        "uq_lgbm_training_jobs_symbol_active",
        "lgbm_training_jobs",
        ["symbol"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_STATUS),
        sqlite_where=sa.text(_ACTIVE_STATUS),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_lgbm_training_jobs_symbol_active",
        table_name="lgbm_training_jobs",
    )
