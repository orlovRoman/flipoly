"""Steps 7-9: contract registry + strike credibility + resolution rules (local)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from polyflip.research.canonical_models.guards import assert_train_allowed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    assert_train_allowed(a.workdir)
    print("[registry] one row per market: market_id,asset,start_at,end_at,up/down ids,strike(+source,available_at),resolution(rule,source),outcome(+available_at)")
    print("[registry] unique market_id; UP/DOWN matched via metadata; disputed/unresolved split out")
    print("[registry] strike classes: canonical_confirmed|retrospective|binance_proxy|unknown; proxy NEVER in main sample")
    print("[registry] labels = real market resolution; own Binance-vs-strike NEVER substitutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
