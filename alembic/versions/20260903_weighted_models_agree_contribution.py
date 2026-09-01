"""Persist the weighted models-agree log-odds contribution."""
from alembic import op
import sqlalchemy as sa

revision = "20260903_weighted_models_agree_contribution"
down_revision = "20260902_weighted_policy_cost_breakdown"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table_name in ("trade_history", "decision_funnel_log"):
        op.add_column(
            table_name,
            sa.Column(
                "weighted_models_agree_contribution_logodds",
                sa.Float(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    for table_name in reversed(("trade_history", "decision_funnel_log")):
        op.drop_column(table_name, "weighted_models_agree_contribution_logodds")
