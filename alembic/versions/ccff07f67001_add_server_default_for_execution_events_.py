"""Add server_default for execution_events.created_at

Revision ID: ccff07f67001
Revises: f2a3b4c5d6e7
Create Date: 2026-08-01 07:28:55.447044+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ccff07f67001'
down_revision: Union[str, None] = 'f2a3b4c5d6e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'execution_events',
        'created_at',
        server_default=sa.text('now()'),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'execution_events',
        'created_at',
        server_default=None,
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
    )
