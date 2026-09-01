"""Persist weighted-policy probabilities, contributions and cost telemetry."""
from alembic import op
import sqlalchemy as sa

revision = "20260901_weighted_policy_telemetry"
down_revision = "20260829_live_session_assets"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("p_market_yes", sa.Float(), True),
    ("p_logreg_yes", sa.Float(), True),
    ("p_lgbm_yes", sa.Float(), True),
    ("weighted_policy_mode", sa.String(length=24), True),
    ("weighted_p_market_yes", sa.Float(), True),
    ("weighted_p_logreg_yes", sa.Float(), True),
    ("weighted_p_lgbm_yes", sa.Float(), True),
    ("weighted_p_final_yes", sa.Float(), True),
    ("weighted_market_weight", sa.Float(), True),
    ("weighted_logreg_weight", sa.Float(), True),
    ("weighted_lgbm_weight", sa.Float(), True),
    ("weighted_mrf_evidence", sa.Float(), True),
    ("weighted_market_contribution_logodds", sa.Float(), True),
    ("weighted_logreg_contribution_logodds", sa.Float(), True),
    ("weighted_lgbm_contribution_logodds", sa.Float(), True),
    ("weighted_mrf_contribution_logodds", sa.Float(), True),
    ("weighted_intercept_contribution_logodds", sa.Float(), True),
    ("weighted_models_agree", sa.Boolean(), True),
    ("weighted_selected_side", sa.String(length=16), True),
    ("weighted_yes_net_ev", sa.Float(), True),
    ("weighted_no_net_ev", sa.Float(), True),
    ("weighted_net_ev_per_share", sa.Float(), True),
    ("weighted_cost_per_share", sa.Float(), True),
    ("weighted_fee_rate", sa.Float(), True),
    ("weighted_fee_exponent", sa.Float(), True),
    ("weighted_fee_per_share", sa.Float(), True),
    ("weighted_slippage_per_share", sa.Float(), True),
    ("weighted_missing_components", sa.String(length=128), True),
    ("weighted_selection_reason", sa.String(length=128), True),
    ("weighted_fee_source", sa.String(length=64), True),
    ("weighted_policy_id", sa.String(length=64), True),
    ("weighted_edge_lower_bound", sa.Float(), True),
    ("weighted_size_multiplier", sa.Float(), True),
    ("weighted_execution_role", sa.String(length=16), True),
    ("weighted_benchmark_json", sa.Text(), True),
)


def upgrade() -> None:
    for table_name in ("trade_history", "decision_funnel_log"):
        for name, column_type, _nullable in _COLUMNS:
            op.add_column(
                table_name,
                sa.Column(name, column_type, nullable=True),
            )


def downgrade() -> None:
    for table_name in reversed(("trade_history", "decision_funnel_log")):
        for name, _column_type, _nullable in reversed(_COLUMNS):
            op.drop_column(table_name, name)
