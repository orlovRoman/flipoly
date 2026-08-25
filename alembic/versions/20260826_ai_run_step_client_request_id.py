"""add client_request_id to ai_run_steps for idempotent proposals

Revision ID: 20260826_ai_run_step_client_request_id
Revises: 20260825_ai_worker_lease_worker_id
"""
from alembic import op
import sqlalchemy as sa

revision = "20260826_ai_run_step_client_request_id"
down_revision = "20260825_ai_worker_lease_worker_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("ai_run_steps")}
    if "client_request_id" not in cols:
        op.add_column("ai_run_steps", sa.Column("client_request_id", sa.String(length=64), nullable=True))
    try:
        op.create_unique_constraint("uix_ai_run_step_client_id", "ai_run_steps", ["run_id", "client_request_id"])
    except Exception:
        pass
    try:
        op.create_index("idx_ai_run_steps_client_id", "ai_run_steps", ["run_id", "client_request_id"])
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_index("idx_ai_run_steps_client_id", table_name="ai_run_steps")
    except Exception:
        pass
    try:
        op.drop_constraint("uix_ai_run_step_client_id", "ai_run_steps", type_="unique")
    except Exception:
        pass
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("ai_run_steps")}
    if "client_request_id" in cols:
        op.drop_column("ai_run_steps", "client_request_id")
