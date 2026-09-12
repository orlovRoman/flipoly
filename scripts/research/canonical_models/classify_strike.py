"""Step 8 helper: classify strike credibility (see build_registry)."""
from __future__ import annotations


def classify(source: str | None, available_before_decision: bool | None) -> str:
    if source == "binance":
        return "binance_proxy"
    if source in ("canonical_confirmed", "retrospective"):
        return source if available_before_decision else "unknown"
    return "unknown"


def main() -> int:
    print("[classify] canonical_confirmed | retrospective | binance_proxy | unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
