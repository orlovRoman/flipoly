"""Step 5: schema/source audit -> source_mapping.md status (local, read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

REQUIRED = ["market_id", "start_at", "end_at", "up_token_id", "down_token_id",
            "strike_value", "strike_source", "resolution_rule", "actual_outcome",
            "outcome_available_at", "quotes", "underlying", "fee"]


def main() -> int:
    print("[audit] required fields:", ", ".join(REQUIRED))
    print("[audit] fill polyflip/research/canonical_models/source_mapping.md from the REAL schema; never guess names")
    print("[audit] collector refs: polyflip/collector/client.py::_canonical_strike, polyflip/collector/resolver.py::extract_final_outcome")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
