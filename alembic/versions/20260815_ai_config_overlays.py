"""Add ai_config_overlays table and expand autonomy_level constraint.

Revision ID: 20260815_ai_config_overlays
Revises: 20260814_ai_experiment_configs
Create Date: 2026-08-15 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260815_ai_config_overlays"
down_revision = "20260814_ai_experiment_configs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create ai_config_overlays table
    op.create_table(
        "ai_config_overlays",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("parent_overlay_id", sa.Integer(), nullable=True),
        sa.Column("scope", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=False),
        sa.Column("changes", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=False),
        sa.Column("base_settings_hash", sa.String(length=64), nullable=False),
        sa.Column("resulting_settings_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="PENDING", nullable=False),
        sa.Column("created_by", sa.String(length=128), server_default="ai_agent", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rollback_payload", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'APPLIED', 'EXPIRED', 'ROLLED_BACK', 'REJECTED')",
            name="ck_ai_config_overlays_status",
        ),
        sa.ForeignKeyConstraint(
            ["parent_overlay_id"],
            ["ai_config_overlays.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ai_optimization_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_ai_overlay_run_status", "ai_config_overlays", ["run_id", "status"], unique=False)
    op.create_index("idx_ai_overlay_expires", "ai_config_overlays", ["expires_at"], unique=False)

    # 2. Update ck_ai_runs_autonomy_level constraint
    op.execute("ALTER TABLE ai_optimization_runs DROP CONSTRAINT IF EXISTS ck_ai_runs_autonomy_level")
    op.create_check_constraint(
        "ck_ai_runs_autonomy_level",
        "ai_optimization_runs",
        "autonomy_level IN ('OBSERVE', 'EXPERIMENT', 'SHADOW', 'AUTONOMOUS_SHADOW', 'AUTONOMOUS_CONFIG', 'LIVE_PROPOSE', 'AUTONOMOUS_LIVE', 'DIRECTED')",
    )


def downgrade() -> None:
    op.drop_index("idx_ai_overlay_expires", table_name="ai_config_overlays")
    op.drop_index("idx_ai_overlay_run_status", table_name="ai_config_overlays")
    op.drop_table("ai_config_overlays")
    op.execute("ALTER TABLE ai_optimization_runs DROP CONSTRAINT IF EXISTS ck_ai_runs_autonomy_level")
    op.create_check_constraint(
        "ck_ai_runs_autonomy_level",
        "ai_optimization_runs",
        "autonomy_level IN ('OBSERVE', 'EXPERIMENT', 'SHADOW', 'LIVE_PROPOSE', 'AUTONOMOUS_SHADOW', 'DIRECTED')",
    )
