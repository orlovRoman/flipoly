"""add worker_id to ai_worker_leases for idempotent claims

Revision ID: 20260825_ai_worker_lease_worker_id
Revises: 20260824_ai_run_llm_snapshot
"""
from alembic import op
import sqlalchemy as sa

revision = "20260825_ai_worker_lease_worker_id"
down_revision = "20260824_widen_alembic_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("ai_worker_leases")}
    if "worker_id" not in cols:
        op.add_column(
            "ai_worker_leases",
            sa.Column("worker_id", sa.String(length=128), nullable=False, server_default="external-ai-research-agent"),
        )
    try:
        op.create_index("idx_ai_worker_leases_worker", "ai_worker_leases", ["worker_id"])
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_index("idx_ai_worker_leases_worker", table_name="ai_worker_leases")
    except Exception:
        pass
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("ai_worker_leases")}
    if "worker_id" in cols:
        op.drop_column("ai_worker_leases", "worker_id")
