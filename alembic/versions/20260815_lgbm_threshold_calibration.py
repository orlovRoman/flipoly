"""Persist threshold and calibration settings for LightGBM experiments."""

from alembic import op
import sqlalchemy as sa


revision = "20260815_lgbm_threshold_calibration"
down_revision = "20260815_ai_agent_run_states"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "lgbm_experiment_configs",
        sa.Column("threshold_params", sa.JSON(), nullable=True),
    )
    op.execute(
        "UPDATE lgbm_experiment_configs "
        "SET threshold_params = '{}' "
        "WHERE threshold_params IS NULL"
    )
    op.alter_column(
        "lgbm_experiment_configs",
        "threshold_params",
        existing_type=sa.JSON(),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("lgbm_experiment_configs", "threshold_params")
