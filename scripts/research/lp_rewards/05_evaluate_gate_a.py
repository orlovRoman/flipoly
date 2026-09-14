"""05_evaluate_gate_a.py

Evaluates Gate A criteria using block bootstrap strictly by full UTC-day.
Determines verdicts: PROCEED_LIVE, PROFITABLE_BELOW_TARGET, TARGET_PLAUSIBLE, TARGET_REJECTED, EDGE_REJECTED.
"""

from decimal import Decimal
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.evaluation import (
    compute_block_bootstrap_ci,
    determine_gate_a_verdict,
)
from polyflip.research.lp_rewards.protocol import load_protocol


def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)
    daily_eval_dir = storage_path / "daily_evaluations"

    if not daily_eval_dir.exists() or not list(daily_eval_dir.glob("*.json")):
        print(f"Notice: No completed daily evaluations found in {daily_eval_dir}.")
        print("Gate A cannot be evaluated without completed daily evaluations.")
        return

    daily_pnls = []
    for f in sorted(daily_eval_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            daily_pnls.append(Decimal(str(data.get("net_pnl", "0.0"))))

    point, lower, upper = compute_block_bootstrap_ci(
        daily_pnls,
        n_bootstrap=protocol.gates.gate_a.bootstrap_samples,
    )
    verdict = determine_gate_a_verdict(
        point,
        lower,
        upper,
        target_rate=protocol.hypothesis.target_r100_calendar,
        total_days=len(daily_pnls),
        min_days=protocol.gates.gate_a.min_calendar_days,
    )

    print(f"Gate A Evaluation Results ({len(daily_pnls)} full UTC days):")
    print(f"  Point Estimate R_100_calendar: ${point:.2f}/day")
    print(f"  95% CI: [${lower:.2f}, ${upper:.2f}] / day")
    print(f"  Target Threshold: ${protocol.hypothesis.target_r100_calendar:.2f}/day")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
