"""
Trade lifecycle helpers and state transition single sources of truth.
"""

from decimal import Decimal
from polyflip.execution.states import ExitReason


def mark_trade_resolved(trade, *, is_win: bool) -> None:
    """
    Единая точка правды для смены состояния сделки при расчете рынка (Settlement).
    ВАЖНО: exit_reason устанавливается ПЕРВЫМ, чтобы не провоцировать ложный warning
    в @validates("position_status").
    """
    trade.exit_reason = ExitReason.SETTLEMENT
    trade.position_status = "RESOLVED_REDEEMABLE" if is_win else "RESOLVED_LOST"
    trade.redemption_status = "PENDING" if is_win else "NOT_REQUIRED"
    if not is_win:
        trade.expected_payout_usdc = Decimal("0")
        if trade.remaining_shares is None or trade.remaining_shares > Decimal("0"):
            trade.remaining_shares = Decimal("0")
