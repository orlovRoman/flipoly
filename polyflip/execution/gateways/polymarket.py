import os
import time
import structlog
from typing import Optional
from decimal import Decimal
from datetime import datetime, timezone

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams
from py_clob_client.constants import POLYGON
from polyflip.execution.contracts import GatewayOrder, ProviderFill, SubmissionResult

logger = structlog.get_logger(__name__)

class PolymarketExecutionGateway:
    name = "POLYMARKET"
    
    def __init__(self):
        self.host = "https://clob.polymarket.com"
        self.chain_id = POLYGON
        self._client_cache: Optional[ClobClient] = None
        self._client_cache_time: float = 0
        self._last_key_hash: str = ""
        self._last_address: str = ""

    def get_client(self) -> Optional[ClobClient]:
        private_key = os.getenv("POLYGON_PRIVATE_KEY")
        address = os.getenv("POLYGON_ADDRESS")
        
        if not private_key or not address:
            return None
            
        import hashlib
        import hmac
        
        now = time.time()
        current_key_hash = hashlib.sha256(private_key.encode()).hexdigest()
        
        if (self._client_cache and 
            (now - self._client_cache_time) < 300 and
            hmac.compare_digest(self._last_key_hash, current_key_hash) and
            self._last_address == address):
            return self._client_cache
            
        try:
            client = ClobClient(
                self.host,
                key=private_key,
                chain_id=self.chain_id
            )
            client.set_creds(client.create_or_derive_creds())
            
            self._client_cache = client
            self._client_cache_time = now
            self._last_key_hash = current_key_hash
            self._last_address = address
            
            return client
        except Exception as e:
            logger.error("failed_to_init_clob_client", error=str(e))
            return None

    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        client = self.get_client()
        if not client:
            return SubmissionResult(
                provider_order_id="",
                status="ERROR: Client not initialized"
            )
            
        try:
            order_args = OrderArgs(
                price=float(order.limit_price),
                size=float(order.requested_shares),
                side=order.side,
                token_id=order.token_id
            )
            
            resp = client.create_and_post_order(order_args, order_type=OrderType.FOK)
            if resp and resp.get("success"):
                return SubmissionResult(
                    provider_order_id=resp.get("orderID", ""),
                    status="SUBMITTED"
                )
            
            err = resp.get("errorMsg") if resp else "Unknown API error"
            return SubmissionResult(
                provider_order_id="",
                status=f"REJECTED: {err}"
            )
        except Exception as e:
            return SubmissionResult(
                provider_order_id="",
                status=f"ERROR: {str(e)}"
            )

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        client = self.get_client()
        if not client:
            return SubmissionResult(
                provider_order_id=provider_order_id,
                status="ERROR: Client not initialized"
            )
            
        try:
            resp = client.get_order(provider_order_id)
            if not resp:
                return SubmissionResult(
                    provider_order_id=provider_order_id,
                    status="UNKNOWN"
                )
            
            status = resp.get("status", "UNKNOWN")
            fills = []
            
            if status == "FILLED":
                filled_shares = Decimal(str(resp.get("size_matched", 0)))
                avg_price = Decimal(str(resp.get("price", 0)))
                if filled_shares > 0:
                    fills.append(ProviderFill(
                        provider_trade_id=provider_order_id,
                        price=avg_price,
                        shares=filled_shares,
                        fee_usdc=Decimal("0"),
                        matched_at=datetime.now(timezone.utc)
                    ))
            
            return SubmissionResult(
                provider_order_id=provider_order_id,
                status=status,
                fills=tuple(fills)
            )
        except Exception as e:
            return SubmissionResult(
                provider_order_id=provider_order_id,
                status=f"ERROR: {str(e)}"
            )

    async def get_balance(self) -> Decimal:
        client = self.get_client()
        if not client:
            return Decimal("0")
        try:
            resp = client.get_balance_allowance(BalanceAllowanceParams(asset_type="collateral"))
            if isinstance(resp, list):
                for b in resp:
                    if b.get("asset_type", "") == "collateral":
                        return Decimal(str(b.get("balance", 0)))
            return Decimal("0")
        except Exception as e:
            logger.error("polymarket_gateway_balance_error", error=str(e))
            return Decimal("0")
