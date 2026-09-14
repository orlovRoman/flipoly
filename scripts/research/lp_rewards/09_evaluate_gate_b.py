"""09_evaluate_gate_b.py

Evaluates Gate B criteria on live calibration data (Days 8-21).
Criteria:
- lower_95(R_100_calendar) >= $3.50/day -> TARGET_CONFIRMED
- upper_95(R_100_calendar) < $3.50/day -> TARGET_NOT_CONFIRMED
- Otherwise -> INCONCLUSIVE
"""

from decimal import Decimal
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.evaluation import determine_gate_b_verdict
from polyflip.research.lp_rewards.protocol import load_protocol


def main():
    protocol = load_protocol()
    print(f"Gate B Evaluator for protocol: {protocol.protocol_id}")
    print(f"Target Confirmation Threshold: ${protocol.hypothesis.target_r100_calendar:.2f}/day (lower 95% bound)")


if __name__ == "__main__":
    main()
