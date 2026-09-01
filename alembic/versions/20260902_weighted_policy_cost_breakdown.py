"""Persist weighted-policy execution cost breakdown."""
from alembic import op
import sqlalchemy as sa


revision = "20260902_weighted_policy_cost_breakdown"
down_revision = "20260901_weighted_policy_telemetry"
branch_labels = None
depends_on = None


_COLUMNS = (
    ("weighted_maker_fee_rate", sa.Float()),
    ("weighted_execution_role", sa.String(length=16)),
    ("weighted_maker_fee_per_share", sa.Float()),
    ("weighted_taker_fee_per_share", sa.Float()),
    ("weighted_spread_per_share", sa.Float()),
    ("weighted_latency_buffer_per_share", sa.Float()),
    ("weighted_expected_execution_price", sa.Float()),
)


def upgrade() -> None:
    for table_name in ("trade_history", "decision_funnel_log"):
        for name, column_type in _COLUMNS:
            op.add_column(table_name, sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    for table_name in reversed(("trade_history", "decision_funnel_log")):
        for name, _column_type in reversed(_COLUMNS):
            op.drop_column(table_name, name)
