"""Persist the LLM provider and model selection for each AI Lab run."""

from alembic import op
import sqlalchemy as sa

revision = "20260822_ai_lab_llm_selection"
down_revision = "mrf_v2_columns_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_optimization_runs")}
    additions = (
        ("llm_provider", sa.String(length=32)),
        ("llm_research_model", sa.String(length=128)),
        ("llm_summary_model", sa.String(length=128)),
    )
    for name, type_ in additions:
        if name not in columns:
            op.add_column("ai_optimization_runs", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_optimization_runs")}
    for name in ("llm_summary_model", "llm_research_model", "llm_provider"):
        if name in columns:
            op.drop_column("ai_optimization_runs", name)

