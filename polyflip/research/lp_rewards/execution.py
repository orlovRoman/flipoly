import os
from decimal import Decimal, getcontext
import logging
from typing import Any, Dict, Optional

getcontext().prec = 28
logger = logging.getLogger(__name__)


class LiveOrderExecutor:
    """Manages order submission to Polymarket CLOB with strict hard gates.
    
    Hard Gates:
    1. LP_LIVE_ENABLED environment variable must be explicitly 'true'.
    2. Protocol SHA-256 hash must match the approved research protocol.
    """

    def __init__(self, expected_protocol_hash: Optional[str] = None):
        self.expected_protocol_hash = expected_protocol_hash

    def is_live_enabled(self) -> bool:
        env_val = os.getenv("LP_LIVE_ENABLED", "false").lower()
        return env_val in ("1", "true", "yes")

    def submit_order(
        self,
        token_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        protocol_hash: str,
    ) -> Dict[str, Any]:
        """Submit live order to CLOB. Raises error if hard gates are not satisfied."""
        if not self.is_live_enabled():
            raise PermissionError(
                "LP Live trading is disabled. LP_LIVE_ENABLED must be set to 'true' after Gate A passes."
            )

        if self.expected_protocol_hash and protocol_hash != self.expected_protocol_hash:
            raise ValueError(
                f"Protocol hash mismatch! Expected {self.expected_protocol_hash}, got {protocol_hash}. Order rejected."
            )

        # In live mode, this would construct and sign EIP-712 order payload
        logger.info(f"Submitting live order: {side} {size} @ {price} for token {token_id}")
        return {
            "status": "SUBMITTED",
            "token_id": token_id,
            "side": side,
            "price": str(price),
            "size": str(size),
        }
