"""split discovery and probe fields for ai_llm_model_catalog

Revision ID: 20260827_ai_llm_catalog_split
Revises: 20260826_ai_run_step_client_request_id
"""
from alembic import op
import sqlalchemy as sa

revision = "20260827_ai_llm_catalog_split"
down_revision = "20260826_ai_run_step_client_request_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("ai_llm_model_catalog")}
    if "is_discovered" not in cols:
        op.add_column("ai_llm_model_catalog", sa.Column("is_discovered", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    if "probe_status" not in cols:
        op.add_column("ai_llm_model_catalog", sa.Column("probe_status", sa.String(length=16), nullable=False, server_default=sa.text("'UNCHECKED'")))
    if "last_checked_at" not in cols:
        op.add_column("ai_llm_model_catalog", sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True))
    # Backfill existing rows: is_discovered true, probe_status UNCHECKED -> if is_available true treat as PASSED? Keep UNCHECKED for now.
    try:
        op.execute(sa.text("UPDATE ai_llm_model_catalog SET is_discovered = true WHERE is_discovered IS NULL"))
    except Exception:
        pass
    try:
        op.execute(sa.text("UPDATE ai_llm_model_catalog SET probe_status = 'UNCHECKED' WHERE probe_status IS NULL"))
    except Exception:
        pass


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("ai_llm_model_catalog")}
    if "last_checked_at" in cols:
        try:
            op.drop_column("ai_llm_model_catalog", "last_checked_at")
        except Exception:
            pass
    if "probe_status" in cols:
        try:
            op.drop_column("ai_llm_model_catalog", "probe_status")
        except Exception:
            pass
    if "is_discovered" in cols:
        try:
            op.drop_column("ai_llm_model_catalog", "is_discovered")
        except Exception:
            pass
