"""Persist raw direction opinions in the decision funnel log."""

from alembic import op
import sqlalchemy as sa


revision = "20260815_funnel_raw_opinion"
down_revision = "20260815_lgbm_threshold_calib"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("decision_funnel_log")}
    additions = {
        "direction_raw_opinion": sa.String(length=16),
        "direction_p_up_raw": sa.Float(),
        "direction_p_down_raw": sa.Float(),
    }
    for name, column_type in additions.items():
        if name not in columns:
            op.add_column(
                "decision_funnel_log",
                sa.Column(name, column_type, nullable=True),
            )


def downgrade() -> None:
    op.drop_column("decision_funnel_log", "direction_p_down_raw")
    op.drop_column("decision_funnel_log", "direction_p_up_raw")
    op.drop_column("decision_funnel_log", "direction_raw_opinion")
