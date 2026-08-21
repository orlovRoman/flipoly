"""Persist the owner token of each durable AI Lab job."""

from alembic import op
import sqlalchemy as sa


revision = "20260820_ai_job_owner_token"
down_revision = "20260820_ai_shadow_active_ask"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_experiment_jobs")}
    if "owner_token" not in columns:
        op.add_column(
            "ai_experiment_jobs",
            sa.Column("owner_token", sa.String(length=128), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_experiment_jobs")}
    if "owner_token" in columns:
        op.drop_column("ai_experiment_jobs", "owner_token")
