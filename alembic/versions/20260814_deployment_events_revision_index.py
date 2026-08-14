"""Add revision_id DESC index on deployment_events.

Revision ID: 20260814_deployment_events_revision_index
Revises: 20260814_deployment_system
Create Date: 2026-08-14 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "20260814_deployment_events_revision_index"
down_revision = "20260814_deployment_system"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_deployment_events_revision_id_desc",
        "deployment_events",
        ["revision_id", sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_deployment_events_revision_id_desc", table_name="deployment_events")
