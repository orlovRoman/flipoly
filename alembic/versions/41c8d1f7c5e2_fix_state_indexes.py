"""fix_state_indexes

Revision ID: 41c8d1f7c5e2
Revises: 32d70d1e31e4
Create Date: 2026-07-27 10:55:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '41c8d1f7c5e2'
down_revision: Union[str, None] = '32d70d1e31e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ACTIVE_OPEN_SQL = """
    intent = 'OPEN' AND state IN (
        'AWAITING_APPROVAL', 'READY', 'CLAIMED', 'SUBMITTING',
        'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
    )
"""

ACTIVE_CLOSE_SQL = """
    intent = 'CLOSE' AND state IN (
        'AWAITING_APPROVAL', 'READY', 'CLAIMED', 'SUBMITTING',
        'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING', 'MANUAL_REVIEW_REQUIRED'
    )
"""

def upgrade() -> None:
    conn = op.get_bind()
    
    # Check for duplicates as requested
    res = conn.execute(sa.text(f"""
        SELECT requested_mode, market_id, count(*) 
        FROM execution_requests 
        WHERE {ACTIVE_OPEN_SQL} 
        GROUP BY requested_mode, market_id HAVING count(*) > 1
    """))
    if len(list(res)) > 0:
        raise Exception("Duplicate active OPEN requests found! Migration aborted.")
        
    res_close = conn.execute(sa.text(f"""
        SELECT requested_mode, trade_history_id, count(*) 
        FROM execution_requests 
        WHERE {ACTIVE_CLOSE_SQL} 
        GROUP BY requested_mode, trade_history_id HAVING count(*) > 1
    """))
    if len(list(res_close)) > 0:
        raise Exception("Duplicate active CLOSE requests found! Migration aborted.")

    op.drop_index("uq_active_open_request", table_name="execution_requests")
    op.create_index(
        "uq_active_open_request", 
        "execution_requests", 
        ["requested_mode", "market_id"], 
        unique=True, 
        postgresql_where=sa.text(ACTIVE_OPEN_SQL),
        sqlite_where=sa.text(ACTIVE_OPEN_SQL)
    )
    
    op.drop_index("uq_active_close_request", table_name="execution_requests")
    op.create_index(
        "uq_active_close_request", 
        "execution_requests", 
        ["requested_mode", "trade_history_id"], 
        unique=True, 
        postgresql_where=sa.text(ACTIVE_CLOSE_SQL),
        sqlite_where=sa.text(ACTIVE_CLOSE_SQL)
    )

def downgrade() -> None:
    pass
