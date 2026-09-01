"""Restore the one-ten-dollar live-order buffer."""

from alembic import op
import sqlalchemy as sa


revision = "20260824_live_minimum_110"
down_revision = "20260824_live_order_minimum_one"
branch_labels = None
depends_on = None


def _set_constraint(minimum: str) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    constraints = {
        c["name"] for c in inspector.get_check_constraints("live_trading_sessions")
    }
    if "ck_live_session_order_amount" in constraints:
        op.drop_constraint(
            "ck_live_session_order_amount",
            "live_trading_sessions",
            type_="check",
        )
    op.create_check_constraint(
        "ck_live_session_order_amount",
        "live_trading_sessions",
        f"""order_amount_usdc IS NULL
        OR (
            order_amount_usdc >= {minimum}
            AND order_amount_usdc <= max_single_order_usdc
            AND order_amount_usdc <= max_total_exposure_usdc
            AND order_amount_usdc <= budget_usdc
        )""",
    )


def upgrade() -> None:
    _set_constraint("1.10")


def downgrade() -> None:
    # Revert only this migration; the previous revision intentionally allows
    # the one-dollar minimum.
    _set_constraint("1.00")
