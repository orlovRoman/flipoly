"""store the immutable LLM selection snapshot on each AI Lab run

Revision ID: 20260824_ai_run_llm_snapshot
Revises: 20260823_ai_llm_model_catalog
"""

from alembic import op
import sqlalchemy as sa


revision = "20260824_ai_run_llm_snapshot"
down_revision = "20260823_ai_llm_model_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_optimization_runs")}
    if "llm_snapshot" not in columns:
        op.add_column(
            "ai_optimization_runs",
            sa.Column("llm_snapshot", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_optimization_runs")}
    if "llm_snapshot" in columns:
        op.drop_column("ai_optimization_runs", "llm_snapshot")
