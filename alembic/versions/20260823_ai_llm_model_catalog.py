"""add ai_llm_model_catalog for dynamic OpenCode model discovery

Revision ID: 20260823_ai_llm_model_catalog
Revises: 20260822_ai_lab_llm_selection
"""

from alembic import op
import sqlalchemy as sa


revision = "20260823_ai_llm_model_catalog"
down_revision = "20260822_ai_lab_llm_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_llm_model_catalog",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=True),
        sa.Column(
            "protocol", sa.String(length=32), nullable=False,
            server_default="responses",
        ),
        sa.Column(
            "supports_structured_output", sa.Boolean(), nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "is_available", sa.Boolean(), nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "discovered_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )
    op.create_unique_constraint(
        "uix_ai_llm_catalog_provider_model",
        "ai_llm_model_catalog",
        ["provider", "model_id"],
    )
    op.create_index(
        "idx_ai_llm_catalog_provider",
        "ai_llm_model_catalog",
        ["provider"],
    )


def downgrade() -> None:
    op.drop_index("idx_ai_llm_catalog_provider", table_name="ai_llm_model_catalog")
    op.drop_constraint(
        "uix_ai_llm_catalog_provider_model", "ai_llm_model_catalog",
        type_="unique",
    )
    op.drop_table("ai_llm_model_catalog")
