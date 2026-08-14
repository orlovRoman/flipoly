"""Add revision_id DESC index on deployment_events.

Revision ID: 20260814_dep_events_idx
Revises: 20260814_ai_worker_leases
"""

from alembic import op
import sqlalchemy as sa

revision = "20260814_dep_events_idx"
down_revision = "20260814_ai_worker_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {item["name"] for item in inspector.get_indexes("deployment_events")}
    if "idx_deployment_events_revision_id_desc" not in existing:
        op.create_index(
            "idx_deployment_events_revision_id_desc",
            "deployment_events",
            ["revision_id", sa.text("id DESC")],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {item["name"] for item in inspector.get_indexes("deployment_events")}
    if "idx_deployment_events_revision_id_desc" in existing:
        op.drop_index(
            "idx_deployment_events_revision_id_desc",
            table_name="deployment_events",
        )
