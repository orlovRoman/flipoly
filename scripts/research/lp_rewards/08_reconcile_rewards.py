"""08_reconcile_rewards.py

Reconciles actual reward payouts received from Polymarket against projections.
"""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import load_protocol
from polyflip.research.lp_rewards.reward_calibration import RewardCalibrator


def main():
    protocol = load_protocol()
    calibrator = RewardCalibrator(
        max_allowed_error_ratio=protocol.gates.gate_b.max_reward_prediction_error
    )
    print(f"Reconciling rewards for protocol: {protocol.protocol_id}")
    print(f"Max allowed prediction error: {protocol.gates.gate_b.max_reward_prediction_error * 100:.0f}%")


if __name__ == "__main__":
    main()
