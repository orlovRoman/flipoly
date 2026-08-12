"""add immutable LightGBM experiment configurations"""
from alembic import op
import sqlalchemy as sa


revision = "20260812_lgbm_configs"
down_revision = "20260812_lgbm_oof"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lgbm_experiment_configs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("asset", sa.String(length=32), nullable=True),
        sa.Column("volatility_regime", sa.String(length=32), nullable=True),
        sa.Column("feature_set", sa.String(length=8), nullable=False),
        sa.Column("feature_set_version", sa.String(length=64), nullable=False),
        sa.Column("model_params", sa.JSON(), nullable=False),
        sa.Column("calibration_params", sa.JSON(), nullable=False),
        sa.Column("backtest_params", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("parent_id", sa.Integer(), sa.ForeignKey("lgbm_experiment_configs.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index(
        "idx_lgbm_experiment_configs_scope",
        "lgbm_experiment_configs",
        ["asset", "volatility_regime"],
    )
    op.create_index(
        "idx_lgbm_experiment_configs_created_at",
        "lgbm_experiment_configs",
        ["created_at"],
    )
    op.create_index(
        "idx_lgbm_experiment_configs_hash",
        "lgbm_experiment_configs",
        ["config_hash"],
    )


def downgrade() -> None:
    op.drop_index("idx_lgbm_experiment_configs_hash", table_name="lgbm_experiment_configs")
    op.drop_index("idx_lgbm_experiment_configs_created_at", table_name="lgbm_experiment_configs")
    op.drop_index("idx_lgbm_experiment_configs_scope", table_name="lgbm_experiment_configs")
    op.drop_table("lgbm_experiment_configs")
