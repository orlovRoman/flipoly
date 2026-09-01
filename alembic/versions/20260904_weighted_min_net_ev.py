"""Persist the effective weighted minimum net EV threshold."""
from alembic import op
import sqlalchemy as sa

revision = "20260904_weighted_min_net_ev"
down_revision = "20260903_weighted_models_agree_contribution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table_name in ("trade_history", "decision_funnel_log"):
        op.add_column(
            table_name,
            sa.Column("weighted_min_net_ev", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    for table_name in reversed(("trade_history", "decision_funnel_log")):
        op.drop_column(table_name, "weighted_min_net_ev")
