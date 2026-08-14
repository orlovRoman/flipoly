"""add autonomous AI Lab worker leases

Revision ID: 20260814_ai_worker_leases
Revises: 20260814_lgbm_active_job_unique
"""

from alembic import op
import sqlalchemy as sa


revision = "20260814_ai_worker_leases"
down_revision = "20260814_lgbm_active_job_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_worker_leases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("owner_token", sa.String(length=128), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_ai_worker_leases_expires",
        "ai_worker_leases",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_ai_worker_leases_expires", table_name="ai_worker_leases")
    op.drop_table("ai_worker_leases")
