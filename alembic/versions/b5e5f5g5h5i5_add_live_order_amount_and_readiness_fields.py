"""add_live_order_amount_and_readiness_fields

Revision ID: b5e5f5g5h5i5
Revises: ccff07f67001
Create Date: 2026-08-01 19:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5e5f5g5h5i5'
down_revision: Union[str, None] = 'ccff07f67001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. order_amount_usdc в live_trading_sessions
    op.add_column(
        "live_trading_sessions",
        sa.Column(
            "order_amount_usdc",
            sa.Numeric(precision=18, scale=6),
            nullable=True,
        ),
    )

    op.create_check_constraint(
        "ck_live_session_order_amount",
        "live_trading_sessions",
        """
        order_amount_usdc IS NULL
        OR (
            order_amount_usdc >= 1.10
            AND order_amount_usdc <= max_single_order_usdc
            AND order_amount_usdc <= max_total_exposure_usdc
            AND order_amount_usdc <= budget_usdc
        )
        """,
    )

    # 2. readiness_checked_at и readiness_success_at в execution_worker_status
    op.add_column(
        "execution_worker_status",
        sa.Column(
            "readiness_checked_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    
    op.add_column(
        "execution_worker_status",
        sa.Column(
            "readiness_success_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("execution_worker_status", "readiness_success_at")
    op.drop_column("execution_worker_status", "readiness_checked_at")
    
    with op.batch_alter_table("live_trading_sessions") as batch_op:
        batch_op.drop_constraint("ck_live_session_order_amount", type_="check")
        batch_op.drop_column("order_amount_usdc")
