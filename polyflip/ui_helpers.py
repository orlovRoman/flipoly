AVAILABLE_DIRECTION_STATUSES = frozenset({
    "OK",
    "READY",
    "DIRECTION_NONE_FALLBACK_LR",
})


def direction_display_value(
    direction_value: str | None,
    direction_status: str | None = None,
    entry_status: str | None = None,
) -> str:
    """Return an explicit UI state without inferring direction from the bet side."""
    if direction_value in {"UP", "DOWN", "NONE"}:
        return direction_value
    if entry_status == "DIRECTION_UNAVAILABLE":
        return "UNAVAILABLE"
    if direction_status and direction_status not in AVAILABLE_DIRECTION_STATUSES:
        return "UNAVAILABLE"
    return "NONE"
