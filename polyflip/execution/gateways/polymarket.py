import os
import structlog
from typing import Optional
from decimal import Decimal
from datetime import datetime, timezone
import asyncio

from polymarket import AsyncSecureClient
from polyflip.execution.contracts import GatewayOrder, TradeExecution, ProviderOrderState, SubmissionResult
from polyflip.execution.gateways.base import GatewayUnavailable

logger = structlog.get_logger(__name__)

class PolymarketExecutionGateway:
    name = "POLYMARKET"
    
    def __init__(self):
        self._client_cache: Optional[AsyncSecureClient] = None
        self._client_lock = asyncio.Lock()

    async def get_client(self) -> Optional[AsyncSecureClient]:
        private_key = os.getenv("POLYGON_PRIVATE_KEY")
        address = os.getenv("POLYGON_ADDRESS")
        
        if not private_key or not address:
            logger.error("missing_polygon_credentials")
            return None
            
        async with self._client_lock:
            if self._client_cache:
                return self._client_cache
                
            try:
                # AsyncSecureClient.create handles setup and gasless deployment
                client = await AsyncSecureClient.create(
                    private_key=private_key,
                    wallet=address
                )
                self._client_cache = client
                return client
            except Exception as e:
                logger.error("failed_to_init_async_secure_client", error=str(e))
                return None

    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")
            
        try:
            amount = float(order.max_spend_usdc) if order.side.upper() == "BUY" and order.max_spend_usdc else float(order.requested_shares)
            
            # Using post_order_with_allowance_recovery which automatically handles collateral allowances
            resp = await client.post_order_with_allowance_recovery(
                token_id=order.token_id,
                price=float(order.limit_price),
                side=order.side.upper(),
                size=amount,
            )
            
            if resp and isinstance(resp, dict) and resp.get("success"):
                return SubmissionResult(
                    provider_order_id=resp.get("orderID", ""),
                    status="SUBMITTED"
                )
            elif resp and hasattr(resp, "success") and getattr(resp, "success"):
                return SubmissionResult(
                    provider_order_id=getattr(resp, "orderID", ""),
                    status="SUBMITTED"
                )
            
            err = getattr(resp, "errorMsg", None) or (resp.get("errorMsg") if isinstance(resp, dict) else "Unknown API error")
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
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")
            
        try:
            resp = await client.get_order(provider_order_id)
            if not resp:
                return SubmissionResult(
                    provider_order_id=provider_order_id,
                    status="UNKNOWN"
                )
            
            # Depending on if resp is a dict or an object
            resp_dict = resp if isinstance(resp, dict) else resp.__dict__
            
            status = resp_dict.get("status", "UNKNOWN")
            fills = []
            
            if status == "FILLED":
                filled_shares = Decimal(str(resp_dict.get("size_matched", 0)))
                avg_price = Decimal(str(resp_dict.get("price", 0)))
                if filled_shares > 0:
                    fills.append(TradeExecution(
                        provider_trade_id=provider_order_id,
                        gateway=self.name,
                        gross_quote_usdc=filled_shares * avg_price,
                        price=avg_price,
                        shares=filled_shares,
                        fee_usdc=Decimal("0"),
                        matched_at=datetime.now(timezone.utc)
                    ))
            
            state = ProviderOrderState(
                provider_order_id=provider_order_id,
                status=status,
                filled_shares=Decimal(str(resp_dict.get("size_matched", 0))),
                filled_usdc=Decimal(str(resp_dict.get("size_matched", 0))) * Decimal(str(resp_dict.get("price", 0))),
                remaining_shares=Decimal("0"), # API doesn't return this directly for filled
                fee_usdc=Decimal("0"),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            return SubmissionResult(
                provider_order_id=provider_order_id,
                status=status,
                state=state,
                fills=tuple(fills)
            )
        except Exception as e:
            return SubmissionResult(
                provider_order_id=provider_order_id,
                status=f"ERROR: {str(e)}"
            )

    async def get_balance(self) -> Decimal:
        client = await self.get_client()
        if not client:
            return Decimal("0")
        try:
            # Assuming get_balance_allowance returns a dict or list similar to old py-clob-client
            resp = await client.get_balance_allowance()
            if isinstance(resp, list):
                for b in resp:
                    if b.get("asset_type", "") == "collateral":
                        return Decimal(str(b.get("balance", 0)))
            elif isinstance(resp, dict):
                return Decimal(str(resp.get("balance", 0)))
            return Decimal("0")
        except Exception as e:
            logger.error("polymarket_gateway_balance_error", error=str(e))
            return Decimal("0")
