"""Merge the AI Lab and MRF production migration heads.

Revision ID: 20260828_merge_ai_lab_mrf_heads
Revises: 20260828_ai_lab_overlay_trade_index, mrf_v3_gate_001
"""

revision = "20260828_merge_ai_lab_mrf_heads"
down_revision = (
    "20260828_ai_lab_overlay_trade_index",
    "mrf_v3_gate_001",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
