"""add_execution_fields

Revision ID: b97b8138716e
Revises: 93bf82b7c9b7
Create Date: 2026-07-26 14:40:46.824620+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b97b8138716e'
down_revision: Union[str, None] = '93bf82b7c9b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add columns to execution_requests
    op.add_column('execution_requests', sa.Column('limit_price', sa.Numeric(precision=38, scale=18), nullable=True))
    op.add_column('execution_requests', sa.Column('max_spend_usdc', sa.Numeric(precision=38, scale=18), nullable=True))
    op.add_column('execution_requests', sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('execution_requests', sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('execution_requests', sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('execution_requests', sa.Column('claimed_by', sa.String(length=100), nullable=True))
    op.add_column('execution_requests', sa.Column('position_version_snapshot', sa.Integer(), nullable=True))
    
    # 2. Add columns to execution_attempts
    op.add_column('execution_attempts', sa.Column('attempt_no', sa.Integer(), nullable=True))
    op.add_column('execution_attempts', sa.Column('provider_order_id', sa.String(length=255), nullable=True))
    op.add_column('execution_attempts', sa.Column('submission_key', sa.String(length=255), nullable=True))
    op.add_column('execution_attempts', sa.Column('provider_status', sa.String(length=50), nullable=True))
    
    # 3. Add columns to execution_fills
    op.add_column('execution_fills', sa.Column('provider_trade_id', sa.String(length=255), nullable=True))
    op.add_column('execution_fills', sa.Column('gateway', sa.String(length=50), nullable=True))
    op.add_column('execution_fills', sa.Column('gross_quote_usdc', sa.Numeric(precision=38, scale=18), nullable=True))
    
    # 4. Add position_version to trade_history
    op.add_column('trade_history', sa.Column('position_version', sa.Integer(), server_default='1', nullable=False))

    # 5. Backfill trade_history_id in execution_requests for OPEN intents
    conn = op.get_bind()
    
    # Check for invalid ones first
    res = conn.execute(sa.text("SELECT id, idempotency_key, trade_history_id FROM execution_requests WHERE intent = 'OPEN'")).fetchall()
    for row in res:
        req_id, idemp_key, th_id = row[0], row[1], row[2]
        if th_id is None:
            if not idemp_key or not idemp_key.startswith("OPEN:"):
                raise RuntimeError(f"Cannot backfill OPEN request {req_id}: invalid idempotency_key '{idemp_key}'")
            try:
                extracted_th_id = int(idemp_key.split(":")[1])
                conn.execute(
                    sa.text("UPDATE execution_requests SET trade_history_id = :th_id WHERE id = :req_id"),
                    {"th_id": extracted_th_id, "req_id": req_id}
                )
            except Exception as e:
                raise RuntimeError(f"Cannot backfill OPEN request {req_id}: failed to parse idempotency_key '{idemp_key}'") from e


def downgrade() -> None:
    # Remove columns from trade_history
    op.drop_column('trade_history', 'position_version')
    
    # Remove columns from execution_fills
    op.drop_column('execution_fills', 'gross_quote_usdc')
    op.drop_column('execution_fills', 'gateway')
    op.drop_column('execution_fills', 'provider_trade_id')
    
    # Remove columns from execution_attempts
    op.drop_column('execution_attempts', 'provider_status')
    op.drop_column('execution_attempts', 'submission_key')
    op.drop_column('execution_attempts', 'provider_order_id')
    op.drop_column('execution_attempts', 'attempt_no')
    
    # Remove columns from execution_requests
    op.drop_column('execution_requests', 'position_version_snapshot')
    op.drop_column('execution_requests', 'claimed_by')
    op.drop_column('execution_requests', 'lease_expires_at')
    op.drop_column('execution_requests', 'claimed_at')
    op.drop_column('execution_requests', 'expires_at')
    op.drop_column('execution_requests', 'max_spend_usdc')
    op.drop_column('execution_requests', 'limit_price')
