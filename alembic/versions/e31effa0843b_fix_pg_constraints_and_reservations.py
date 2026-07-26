"""fix_pg_constraints_and_reservations

Revision ID: e31effa0843b
Revises: e3c38eb8e41a
Create Date: 2026-07-26 17:56:43.083855+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e31effa0843b'
down_revision: Union[str, None] = 'e3c38eb8e41a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    constraints = {c["name"] for c in inspector.get_check_constraints("execution_requests")}
    
    # 1. Safely drop old check if it exists
    if "ck_execution_request_trade_reference" in constraints:
        op.drop_constraint("ck_execution_request_trade_reference", "execution_requests", type_="check")
        
    # 2. Add the correct check if it's missing
    if "ck_execution_request_intent" not in constraints:
        op.create_check_constraint(
            "ck_execution_request_intent", "execution_requests", "intent IN ('OPEN', 'CLOSE')"
        )
        
    # 3. Clean up invalid exposure_reservations and enforce NOT NULL
    # Delete invalid reservations without a request_id
    conn.execute(sa.text("DELETE FROM exposure_reservations WHERE request_id IS NULL"))
    
    # Restore active reservations based on active execution requests
    conn.execute(sa.text("""
        INSERT INTO exposure_reservations (id, request_id, trade_history_id, market_id, amount_usdc, expires_at, created_at)
        SELECT 
            gen_random_uuid(), 
            r.id, 
            r.trade_history_id, 
            r.market_id, 
            r.target_amount_usdc, 
            COALESCE(r.expires_at, r.created_at + interval '60 seconds'), 
            r.created_at
        FROM execution_requests r
        WHERE r.intent = 'OPEN' 
          AND r.state IN ('READY', 'CLAIMED', 'SUBMITTING', 'ACCEPTED', 'UNKNOWN', 'PARTIALLY_FILLED', 'RECONCILING')
          AND NOT EXISTS (SELECT 1 FROM exposure_reservations e WHERE e.request_id = r.id)
    """))
    
    op.alter_column(
        "exposure_reservations",
        "request_id",
        existing_type=sa.UUID(),
        nullable=False
    )

def downgrade() -> None:
    op.alter_column(
        "exposure_reservations",
        "request_id",
        existing_type=sa.UUID(),
        nullable=True
    )
