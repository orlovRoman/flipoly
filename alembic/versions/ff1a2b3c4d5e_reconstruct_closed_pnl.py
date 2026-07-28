"""Reconstruct realized_pnl_usdc for CLOSED positions where it is NULL or zero.

Revision ID: ff1a2b3c4d5e
Revises: b3c4d5e6f7a8
Create Date: 2026-07-28

Formula:
    realized_pnl_usdc = (entry_filled_shares - remaining_shares) * close_price
                        - entry_cost_usdc

Prerequisites:
    - position_status column exists (migration c4df36df0c84)
    - entry_filled_shares, entry_cost_usdc, remaining_shares, realized_pnl_usdc
      columns exist (migration b1c2d3e4f5a6)
    - close_price column exists

Safety:
    - Only touches rows where position_status = 'CLOSED'
    - Only touches rows where realized_pnl_usdc IS NULL OR = 0
    - Requires all four source columns to be non-NULL
    - close_price::numeric cast avoids mixed-type arithmetic error in PG
    - downgrade() is a no-op: reconstructed values are derived data,
      reverting them would just re-zero already-backfilled rows.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'ff1a2b3c4d5e'
down_revision = 'b3c4d5e6f7a8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE trade_history
        SET realized_pnl_usdc = ROUND(
            (entry_filled_shares - remaining_shares)
            * close_price::numeric
            - entry_cost_usdc,
            18
        )
        WHERE
            position_status = 'CLOSED'
            AND (realized_pnl_usdc IS NULL OR realized_pnl_usdc = 0)
            AND close_price       IS NOT NULL
            AND entry_filled_shares IS NOT NULL
            AND remaining_shares    IS NOT NULL
            AND entry_cost_usdc     IS NOT NULL
    """)


def downgrade() -> None:
    # Reconstructed values are derived data — downgrade is intentionally a no-op.
    # Re-zeroing valid PnL would corrupt accounting.
    pass
