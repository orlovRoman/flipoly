"""Persist raw direction opinions in the decision funnel log."""

from alembic import op
import sqlalchemy as sa


revision = "20260815_funnel_raw_opinion"
down_revision = "20260815_lgbm_threshold_calib"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "decision_funnel_log",
        sa.Column("direction_raw_opinion", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "decision_funnel_log",
        sa.Column("direction_p_up_raw", sa.Float(), nullable=True),
    )
    op.add_column(
        "decision_funnel_log",
        sa.Column("direction_p_down_raw", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("decision_funnel_log", "direction_p_down_raw")
    op.drop_column("decision_funnel_log", "direction_p_up_raw")
    op.drop_column("decision_funnel_log", "direction_raw_opinion")
