"""add mrf columns to decision_funnel_log

Revision ID: mrf_v2_columns_001
Revises: 20260820_ai_job_owner_token
Create Date: 2026-08-21
"""
from alembic import op
import sqlalchemy as sa

revision = "mrf_v2_columns_001"
down_revision = "20260820_ai_job_owner_token"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns(table)}
    return column in cols


def upgrade() -> None:
    table = "decision_funnel_log"
    columns = [
        ("mrf_mode", sa.String(16)),
        ("mrf_phase", sa.String(32)),
        ("mrf_asset_phase", sa.String(32)),
        ("mrf_strength", sa.Float()),
        ("mrf_confidence", sa.Float()),
        ("mrf_multiplier", sa.Float()),
        ("mrf_applied", sa.Boolean()),
        ("mrf_evaluated", sa.Boolean()),
        ("mrf_as_of", sa.DateTime(timezone=True)),
        ("mrf_failure_reason", sa.String(256)),
        ("mrf_audit_json", sa.Text()),
        ("mrf_original_action", sa.String(16)),
        ("mrf_original_bet", sa.Float()),
        ("mrf_final_action", sa.String(16)),
        ("mrf_final_bet", sa.Float()),
    ]
    for col_name, col_type in columns:
        if not _column_exists(table, col_name):
            op.add_column(table, sa.Column(col_name, col_type, nullable=True))


def downgrade() -> None:
    table = "decision_funnel_log"
    columns = [
        "mrf_final_bet",
        "mrf_final_action",
        "mrf_original_bet",
        "mrf_original_action",
        "mrf_audit_json",
        "mrf_failure_reason",
        "mrf_as_of",
        "mrf_evaluated",
        "mrf_applied",
        "mrf_multiplier",
        "mrf_confidence",
        "mrf_strength",
        "mrf_asset_phase",
        "mrf_phase",
        "mrf_mode",
    ]
    for col_name in columns:
        if _column_exists(table, col_name):
            op.drop_column(table, col_name)
