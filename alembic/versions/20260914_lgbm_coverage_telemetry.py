"""Add explicit LightGBM evaluation and application attribution.

``direction_model_*`` predates the distinction and remains populated as the
legacy funnel alias.  The new columns let analytics answer separately whether
LightGBM ran and whether its UP/DOWN decision was used by the active policy.
"""

from alembic import op
import sqlalchemy as sa


revision = "20260914_lgbm_coverage_telemetry"
down_revision = "20260913_canonical_paper_fee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "decision_funnel_log",
        sa.Column("evaluated_model_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "decision_funnel_log",
        sa.Column("evaluated_model_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "decision_funnel_log",
        sa.Column("applied_direction_model_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "decision_funnel_log",
        sa.Column("applied_direction_model_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("decision_funnel_log", "applied_direction_model_version")
    op.drop_column("decision_funnel_log", "applied_direction_model_key")
    op.drop_column("decision_funnel_log", "evaluated_model_version")
    op.drop_column("decision_funnel_log", "evaluated_model_key")
