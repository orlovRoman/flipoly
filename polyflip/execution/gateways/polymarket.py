import os
import structlog
from typing import Optional
from decimal import Decimal
from datetime import datetime, timezone
import asyncio

from polymarket import AsyncSecureClient
from polyflip.execution.contracts import GatewayOrder, TradeExecution, ProviderOrderState, SubmissionResult, GatewayUnavailable, GatewayReadiness, BalanceResult

logger = structlog.get_logger(__name__)

class PolymarketExecutionGateway:
    name = "POLYMARKET"
    
    def __init__(self, private_key: str, wallet_address: str, host: str):
        self._private_key = private_key
        self._wallet_address = wallet_address
        self._host = host
        self._client_cache: Optional[AsyncSecureClient] = None
        self._client_lock = asyncio.Lock()

    async def get_client(self) -> Optional[AsyncSecureClient]:
        if not self._private_key or not self._wallet_address:
            logger.error("missing_polygon_credentials")
            return None
            
        async with self._client_lock:
            if self._client_cache:
                return self._client_cache
                
            try:
                client = await AsyncSecureClient.create(
                    private_key=self._private_key,
                    wallet=self._wallet_address
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
            if order.side.upper() == "BUY":
                amount_limit = float(order.max_spend_usdc) if order.max_spend_usdc else 0.0
                resp = await client.place_market_order(
                    token_id=order.token_id,
                    side="BUY",
                    amount=amount_limit,
                    max_spend=amount_limit,
                    max_price=float(order.limit_price),
                    order_type="FAK",
                )
            else:
                resp = await client.place_market_order(
                    token_id=order.token_id,
                    side="SELL",
                    shares=float(order.requested_shares),
                    min_price=float(order.limit_price),
                    order_type="FAK",
                )
            
            if not getattr(resp, "ok", False):
                return SubmissionResult(
                    provider_order_id="",
                    status="REJECTED",
                    rejection_code=getattr(resp, "code", ""),
                    error_message=getattr(resp, "message", "Unknown rejection"),
                )
                
            return SubmissionResult(
                provider_order_id=getattr(resp, "order_id", ""),
                status=getattr(resp, "status", "UNKNOWN").upper(),
                provider_trade_ids=getattr(resp, "trade_ids", [])
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
            resp = await client.get_order(order_id=provider_order_id)
            if not resp:
                return SubmissionResult(
                    provider_order_id=provider_order_id,
                    status="UNKNOWN"
                )
            
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
                remaining_shares=Decimal("0"),
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

    async def get_readiness(
        self, conditional_token_ids: tuple[str, ...] = ()
    ) -> GatewayReadiness:
        credentials_loaded = bool(self._private_key and self._wallet_address)
        client = await self.get_client()
        client_initialized = client is not None
        
        readiness = GatewayReadiness(
            ready=False,
            gateway=self.name,
            wallet_address=self._wallet_address,
            balance=None,
            credentials_loaded=credentials_loaded,
            client_initialized=client_initialized,
            collateral_allowance_ready=False,
            conditional_allowance_ready=None,
            checked_at=datetime.now(timezone.utc)
        )
        
        if not client:
            readiness.error_message = "Polymarket client not initialized"
            return readiness
            
        try:
            resp = await client.get_balance_allowance(asset_type="COLLATERAL")
            raw_balance = Decimal("0")
            allowances = {}
            
            if hasattr(resp, "balance"):
                raw_balance = Decimal(str(getattr(resp, "balance", 0)))
                allowances = getattr(resp, "allowances", {})
                
            balance_usdc = raw_balance / Decimal("1000000")
            collateral_ready = False
            
            parsed_allowances = {}
            for k, v in allowances.items():
                amt = Decimal(str(v)) / Decimal("1000000")
                parsed_allowances[k] = amt
                if amt > 0:
                    collateral_ready = True
            
            balance_result = BalanceResult(
                balance_usdc=balance_usdc,
                collateral_allowances=parsed_allowances,
                conditional_allowances_checked=0,
                checked_at=datetime.now(timezone.utc),
                raw_asset_type="COLLATERAL"
            )
            
            readiness.balance = balance_result
            readiness.collateral_allowance_ready = collateral_ready
            
            readiness.ready = credentials_loaded and client_initialized and collateral_ready
            return readiness
            
        except Exception as e:
            logger.error("polymarket_gateway_readiness_error", error=str(e))
            readiness.error_message = str(e)
            return readiness
