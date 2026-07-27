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
