"""Add MRF v3 veto-gate telemetry to decision funnel logs.

Revision ID: mrf_v3_gate_001
Revises: 20260824_live_minimum_110
"""

from alembic import op
import sqlalchemy as sa


revision = "mrf_v3_gate_001"
down_revision = "20260824_live_minimum_110"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return column in {item["name"] for item in inspector.get_columns(table)}


def upgrade() -> None:
    table = "decision_funnel_log"
    columns = (
        ("mrf_policy_version", sa.Integer()),
        ("mrf_regime_evidence", sa.Float()),
        ("mrf_gate_threshold", sa.Float()),
        ("mrf_edge_margin", sa.Float()),
        ("mrf_gate_would_block", sa.Boolean()),
        ("mrf_gate_reason", sa.String(128)),
    )
    for name, column_type in columns:
        if not _column_exists(table, name):
            op.add_column(table, sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    table = "decision_funnel_log"
    for name in (
        "mrf_gate_reason",
        "mrf_gate_would_block",
        "mrf_edge_margin",
        "mrf_gate_threshold",
        "mrf_regime_evidence",
        "mrf_policy_version",
    ):
        if _column_exists(table, name):
            op.drop_column(table, name)
