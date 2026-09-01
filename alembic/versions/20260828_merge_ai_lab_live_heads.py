"""Merge the AI Lab and live-order migration heads.

Revision ID: 20260828_merge_ai_lab_live_heads
Revises: 20260824_live_minimum_110, 20260828_ai_lab_overlay_trade_trace
"""

revision = "20260828_merge_ai_lab_live_heads"
down_revision = (
    "20260824_live_minimum_110",
    "20260828_ai_lab_overlay_trade_trace",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
