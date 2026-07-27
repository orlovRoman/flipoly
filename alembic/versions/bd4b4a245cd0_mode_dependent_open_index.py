"""mode_dependent_open_index

Revision ID: bd4b4a245cd0
Revises: e31effa0843b
Create Date: 2026-07-27 00:42:04.417678+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bd4b4a245cd0'
down_revision: Union[str, None] = 'e31effa0843b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ACTIVE_SQL = """
    intent = 'OPEN' AND state IN (
        'READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED',
        'UNKNOWN', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
    )
"""

def upgrade() -> None:
    op.drop_index("uq_active_open_request", table_name="execution_requests")
    op.create_index(
        "uq_active_open_request", 
        "execution_requests", 
        ["requested_mode", "market_id"], 
        unique=True, 
        postgresql_where=sa.text(ACTIVE_SQL)
    )

def downgrade() -> None:
    op.drop_index("uq_active_open_request", table_name="execution_requests")
    op.create_index(
        "uq_active_open_request", 
        "execution_requests", 
        ["market_id"], 
        unique=True, 
        postgresql_where=sa.text(
            "intent = 'OPEN' AND state IN "
            "('READY','CLAIMED','SUBMITTING','ACCEPTED',"
            "'UNKNOWN','PARTIALLY_FILLED','RECONCILING')"
        )
    )
