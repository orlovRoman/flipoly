"""store full LightGBM training job tracebacks

Revision ID: 20260813_lgbm_job_traceback
Revises: 20260813_lgbm_training_jobs
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_lgbm_job_traceback"
down_revision = "20260813_lgbm_training_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "lgbm_training_jobs",
        sa.Column("error_traceback", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("lgbm_training_jobs", "error_traceback")
