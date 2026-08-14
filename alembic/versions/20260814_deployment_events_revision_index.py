"""add composite index for deployment_events revision queries

Revision ID: 20260814_deployment_events_revision_index
Revises: 20260814_ai_worker_leases
"""

from alembic import op
import sqlalchemy as sa


revision = "20260814_deployment_events_revision_index"
down_revision = "20260814_ai_worker_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_deployment_events_revision_id_desc",
        "deployment_events",
        ["revision_id", sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_deployment_events_revision_id_desc",
        table_name="deployment_events",
    )
