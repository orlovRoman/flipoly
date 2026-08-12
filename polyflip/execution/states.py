"""
Central definition of execution state machine.
"""

REQUEST_HOLD_STATES = frozenset(
    {
        "AWAITING_APPROVAL",
        "READY",
        "CLAIMED",
        "SUBMITTING",
        "ACCEPTED",  # legacy
        "UNKNOWN",  # legacy
        "PARTIALLY_FILLED",  # legacy, пока не выполнен backfill
        "RECONCILING",
        "MANUAL_REVIEW_REQUIRED",
    }
)

RECONCILABLE_REQUEST_STATES = frozenset(
    {
        "SUBMITTING",
        "ACCEPTED",
        "UNKNOWN",
        "PARTIALLY_FILLED",
        "RECONCILING",
    }
)

SUCCESS_TERMINAL_STATES = frozenset(
    {
        "FILLED",
        "PARTIALLY_FILLED_FINAL",
    }
)

FAILURE_TERMINAL_STATES = frozenset(
    {
        "REJECTED",
        "EXPIRED",
        "CANCELED",
        "MANUAL_REVIEW_FAILED",
    }
)

TERMINAL_REQUEST_STATES = SUCCESS_TERMINAL_STATES | FAILURE_TERMINAL_STATES

ACTIVE_REQUEST_STATES = REQUEST_HOLD_STATES

# Состояния, которые реально блокируют постановку нового OPEN/CLOSE.
# MANUAL_REVIEW_REQUIRED исключён: воркер его не обрабатывает повторно,
# поэтому он не должен мешать новым запросам для того же маркета.
BLOCKING_REQUEST_STATES = REQUEST_HOLD_STATES - {"MANUAL_REVIEW_REQUIRED"}

ACTIVE_POSITION_STATES = frozenset(
    {
        "OPENING",
        "OPEN",
        "PARTIALLY_CLOSED",
        "EXIT_REQUESTED",
        "CLOSING",
        # LIVE-позиции после разрешения рынка — ожидают on-chain redemption
        "RESOLVED_REDEEMABLE",
    }
)

FINAL_POSITION_STATES = frozenset(
    {
        "CLOSED",
        "RESOLVED_REDEEMABLE",
        "RESOLVED_LOST",
        "RESOLVED_WON",
        "REDEEMED",
    }
)


class ExitReason:
    SETTLEMENT = "SETTLEMENT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MANUAL = "MANUAL"
    LIQUIDATION = "LIQUIDATION"

    ALL = frozenset({SETTLEMENT, STOP_LOSS, TAKE_PROFIT, MANUAL, LIQUIDATION})

