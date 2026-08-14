"""add the AI Lab experiment configuration table

Revision ID: 20260814_ai_experiment_configs
Revises: 20260814_schema_compat
"""

from alembic import op
import sqlalchemy as sa


revision = "20260814_ai_experiment_configs"
down_revision = "20260814_schema_compat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "ai_experiment_configs" not in inspector.get_table_names():
        op.create_table(
            "ai_experiment_configs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("asset", sa.String(length=32), nullable=True),
            sa.Column("regime", sa.String(length=32), nullable=True),
            sa.Column("model_family", sa.String(length=32), nullable=False),
            sa.Column("feature_set", sa.String(length=32), nullable=False),
            sa.Column("feature_pipeline_version", sa.String(length=64), nullable=False),
            sa.Column("model_params", sa.JSON(), nullable=False),
            sa.Column("strategy_params", sa.JSON(), nullable=False),
            sa.Column("backtest_params", sa.JSON(), nullable=False),
            sa.Column("config_hash", sa.String(length=64), nullable=False, unique=True),
            sa.Column(
                "parent_id",
                sa.Integer(),
                sa.ForeignKey("ai_experiment_configs.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("created_by", sa.String(length=128), nullable=False, server_default="system"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index(
            "idx_ai_configs_asset_regime",
            "ai_experiment_configs",
            ["asset", "regime"],
        )
        op.create_index(
            "idx_ai_configs_model_family",
            "ai_experiment_configs",
            ["model_family"],
        )


def downgrade() -> None:
    op.drop_index("idx_ai_configs_model_family", table_name="ai_experiment_configs")
    op.drop_index("idx_ai_configs_asset_regime", table_name="ai_experiment_configs")
    op.drop_table("ai_experiment_configs")
