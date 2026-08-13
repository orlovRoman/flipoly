"""persist audit records for invalid AI Lab executor steps

Revision ID: 20260813_ai_step_audit
Revises: 20260813_ai_optimizer
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_ai_step_audit"
down_revision = "20260813_ai_optimizer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_step_audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "step_id",
            sa.Integer(),
            sa.ForeignKey("ai_run_steps.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("config_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_ai_step_audit_run_created",
        "ai_step_audit_logs",
        ["run_id", "created_at"],
    )
    op.create_index(
        "idx_ai_step_audit_code",
        "ai_step_audit_logs",
        ["error_code", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_ai_step_audit_code", table_name="ai_step_audit_logs")
    op.drop_index("idx_ai_step_audit_run_created", table_name="ai_step_audit_logs")
    op.drop_table("ai_step_audit_logs")
