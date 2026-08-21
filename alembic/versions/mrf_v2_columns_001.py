"""add mrf columns to decision_funnel_log

Revision ID: mrf_v2_columns_001
Revises: <head>
Create Date: 2026-08-21
"""
from alembic import op
import sqlalchemy as sa

revision = "mrf_v2_columns_001"
down_revision = ""
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("decision_funnel_log", sa.Column("mrf_mode", sa.String(16), nullable=True))
    op.add_column("decision_funnel_log", sa.Column("mrf_phase", sa.String(32), nullable=True))
    op.add_column("decision_funnel_log", sa.Column("mrf_asset_phase", sa.String(32), nullable=True))
    op.add_column("decision_funnel_log", sa.Column("mrf_strength", sa.Float(), nullable=True))
    op.add_column("decision_funnel_log", sa.Column("mrf_confidence", sa.Float(), nullable=True))
    op.add_column("decision_funnel_log", sa.Column("mrf_multiplier", sa.Float(), nullable=True))
    op.add_column("decision_funnel_log", sa.Column("mrf_applied", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("decision_funnel_log", "mrf_applied")
    op.drop_column("decision_funnel_log", "mrf_multiplier")
    op.drop_column("decision_funnel_log", "mrf_confidence")
    op.drop_column("decision_funnel_log", "mrf_strength")
    op.drop_column("decision_funnel_log", "mrf_asset_phase")
    op.drop_column("decision_funnel_log", "mrf_phase")
    op.drop_column("decision_funnel_log", "mrf_mode")
