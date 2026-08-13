"""add durable LightGBM training jobs

Revision ID: 20260813_lgbm_training_jobs
Revises: merge_all_heads
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_lgbm_training_jobs"
down_revision = "merge_all_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lgbm_training_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("interval", sa.String(length=8), nullable=False, server_default="15m"),
        sa.Column("feature_set", sa.String(length=8), nullable=False),
        sa.Column("activate_after_train", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "experiment_config_id",
            sa.Integer(),
            sa.ForeignKey("lgbm_experiment_configs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="QUEUED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_pid", sa.Integer(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'SUCCESS', 'FAILED')",
            name="ck_lgbm_training_jobs_status",
        ),
    )
    op.create_index(
        "idx_lgbm_training_jobs_status_created",
        "lgbm_training_jobs",
        ["status", "created_at"],
    )
    op.create_index(
        "idx_lgbm_training_jobs_symbol_created",
        "lgbm_training_jobs",
        ["symbol", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_lgbm_training_jobs_symbol_created", table_name="lgbm_training_jobs")
    op.drop_index("idx_lgbm_training_jobs_status_created", table_name="lgbm_training_jobs")
    op.drop_table("lgbm_training_jobs")
