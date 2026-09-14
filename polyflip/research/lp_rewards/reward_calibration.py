import logging
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple
import httpx

getcontext().prec = 28
logger = logging.getLogger(__name__)


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
        """Record reward observation and return relative error ratio strictly relative to actual payout.
        
        Formula: abs(projected - actual) / actual
        """
        if actual_reward <= Decimal("0.0"):
            error_ratio = Decimal("0.0") if projected_reward <= Decimal("0.0") else Decimal("1.0")
        else:
            error_ratio = abs(projected_reward - actual_reward) / actual_reward

        self.history.append({
            "condition_id": condition_id,
            "projected": projected_reward,
            "actual": actual_reward,
            "error_ratio": error_ratio,
        })
        return error_ratio

    async def fetch_actual_user_rewards(
        self,
        wallet_address: str,
        client: Optional[httpx.AsyncClient] = None,
        base_url: str = "https://clob.polymarket.com",
    ) -> List[Dict[str, Any]]:
        """Fetch actual user rewards history from Polymarket CLOB rewards API."""
        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=30.0)
            should_close = True

        try:
            url = f"{base_url}/rewards/user"
            resp = await client.get(url, params={"address": wallet_address})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("data") or data.get("rewards") or [data]
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch actual rewards for wallet {wallet_address}: {e}")
            return []
        finally:
            if should_close:
                await client.aclose()

    def evaluate_gate_b_accuracy(self) -> Tuple[bool, Decimal]:
        """Check if mean relative prediction error is within the 30% threshold."""
        if not self.history:
            return False, Decimal("1.0")

        mean_error = sum((h["error_ratio"] for h in self.history), Decimal("0.0")) / Decimal(str(len(self.history)))
        passes = mean_error <= self.max_allowed_error_ratio
        return passes, mean_error
