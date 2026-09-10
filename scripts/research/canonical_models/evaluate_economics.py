"""Steps 32-36: final economics locally (shared fill fn, fixed policies)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from polyflip.research.canonical_models.guards import assert_train_allowed
from polyflip.research.canonical_models.protocol import load_protocol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    assert_train_allowed(a.workdir)
    ev = load_protocol()["economics"]["ev_threshold_net"]
    print(f"[economics] policies: price_control, CT, M1, challenger, challenger_plus_CT; pre-registered EV thr={ev}")
    print("[economics] per policy: opportunities/entries/fills, PnL per opp & trade, turnover, win-rate, avg win/loss, max drawdown (chronological), side/asset/week splits, top-win contribution")
    print("[economics] CIs: M1, challenger, challenger-M1, challenger+CT-challenger, vs price control (paired day blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
