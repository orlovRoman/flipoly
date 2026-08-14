"""Allow activated deployment revisions to be superseded by a newer revision.

Revision ID: 20260814_deployment_superseded_status
Revises: 20260814_deployment_events_revision_index
"""

from alembic import op


revision = "20260814_deployment_superseded_status"
down_revision = "20260814_deployment_events_revision_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A manifest hash identifies content, but the same content may be
    # proposed again after rollback with a new revision key/parent.
    op.execute(
        "ALTER TABLE deployment_revisions "
        "DROP CONSTRAINT IF EXISTS deployment_revisions_manifest_hash_key"
    )
    op.drop_constraint(
        "ck_deployment_revisions_status",
        "deployment_revisions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_deployment_revisions_status",
        "deployment_revisions",
        "status IN ('DRAFT', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', "
        "'SUPERSEDED', 'REJECTED', 'ROLLED_BACK')",
    )


def downgrade() -> None:
    op.create_unique_constraint(
        "deployment_revisions_manifest_hash_key",
        "deployment_revisions",
        ["manifest_hash"],
    )
    op.execute(
        "UPDATE deployment_revisions "
        "SET status = 'ROLLED_BACK' WHERE status = 'SUPERSEDED'"
    )
    op.drop_constraint(
        "ck_deployment_revisions_status",
        "deployment_revisions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_deployment_revisions_status",
        "deployment_revisions",
        "status IN ('DRAFT', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', "
        "'REJECTED', 'ROLLED_BACK')",
    )
