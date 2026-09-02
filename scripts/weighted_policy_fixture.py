#!/usr/bin/env python3
"""Extract a small resolved real-market regression fixture from a benchmark export."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyflip.trading.weighted_benchmark import MarketObservation, deduplicate_observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--allow-smaller", action="store_true")
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = raw.get("observations", raw) if isinstance(raw, dict) else raw
    observations = [
        MarketObservation.from_mapping(item)
        for item in rows
        if isinstance(item, dict)
    ]
    resolved = [
        item for item in deduplicate_observations(observations)
        if item.outcome_yes is not None
    ]
    size = max(1, int(args.size))
    if len(resolved) < size and not args.allow_smaller:
        raise SystemExit(
            f"fixture needs {size} resolved rows, found {len(resolved)}"
        )
    selected = resolved[:size]
    legacy_lines = "\n".join(
        f"{item.market_id}|{item.timestamp.isoformat()}|{item.legacy_action or 'SKIP'}"
        for item in selected
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_size_requested": size,
        "fixture_size": len(selected),
        # This digest makes the captured legacy decisions an explicit
        # regression contract without changing the runtime decision path.
        "legacy_decision_fingerprint": hashlib.sha256(
            legacy_lines.encode("utf-8")
        ).hexdigest(),
        "observations": [item.as_dict() for item in selected],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
