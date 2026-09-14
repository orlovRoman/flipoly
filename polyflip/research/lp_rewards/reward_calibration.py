from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple

getcontext().prec = 28


class RewardCalibrator:
    """Reconciles projected rewards with actual rewards received from Polymarket."""

    def __init__(self, max_allowed_error_ratio: Decimal = Decimal("0.30")):
        self.max_allowed_error_ratio = max_allowed_error_ratio
        self.history: List[Dict[str, Decimal]] = []

    def record_observation(
        self,
        condition_id: str,
        projected_reward: Decimal,
        actual_reward: Decimal,
    ) -> Decimal:
        """Record reward observation and return relative error ratio."""
        if projected_reward <= Decimal("0.0"):
            error_ratio = Decimal("0.0") if actual_reward <= Decimal("0.0") else Decimal("1.0")
        else:
            error_ratio = abs(actual_reward - projected_reward) / projected_reward

        self.history.append({
            "projected": projected_reward,
            "actual": actual_reward,
            "error_ratio": error_ratio,
        })
        return error_ratio

    def evaluate_gate_b_accuracy(self) -> Tuple[bool, Decimal]:
        """Check if mean relative prediction error is within the 30% threshold."""
        if not self.history:
            return False, Decimal("1.0")

        mean_error = sum((h["error_ratio"] for h in self.history), Decimal("0.0")) / Decimal(str(len(self.history)))
        passes = mean_error <= self.max_allowed_error_ratio
        return passes, mean_error
