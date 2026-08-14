"""Add optional runtime setting descriptions.

Revision ID: 20260814_rt_settings_desc
Revises: 20260814_dep_superseded
"""

from alembic import op


revision = "20260814_rt_settings_desc"
down_revision = "20260814_dep_superseded"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE runtime_settings "
        "ADD COLUMN IF NOT EXISTS description VARCHAR(512)"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE runtime_settings "
        "DROP COLUMN IF EXISTS description"
    )
