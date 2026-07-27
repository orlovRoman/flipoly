import os
import structlog
from typing import Optional
from decimal import Decimal
from datetime import datetime, timezone
import asyncio

from polymarket import AsyncSecureClient
from polymarket.errors import TimeoutError as SettlementTimeoutError, TransactionFailedError
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
                amount_limit = str(order.max_spend_usdc) if order.max_spend_usdc else "0"
                resp = await client.place_market_order(
                    token_id=order.token_id,
                    side="BUY",
                    amount=amount_limit,
                    max_spend=amount_limit,
                    max_price=str(order.limit_price),
                    order_type="FAK",
                )
            else:
                resp = await client.place_market_order(
                    token_id=order.token_id,
                    side="SELL",
                    shares=str(order.requested_shares),
                    min_price=str(order.limit_price),
                    order_type="FAK",
                )
            
            if not getattr(resp, "ok", False):
                return SubmissionResult(
                    accepted=False,
                    provider_status="REJECTED",
                    rejection_code=str(getattr(resp, "code", "")),
                    error_message=getattr(resp, "message", "Unknown rejection"),
                )
                
            settlement_state = "PENDING"
            transaction_hashes: tuple[str, ...] = ()
            trade_ids = getattr(resp, "trade_ids", [])
            
            if trade_ids:
                try:
                    hashes = await client.wait_for_order_fill_settlement(
                        resp, timeout_s=30
                    )
                    transaction_hashes = tuple(map(str, hashes))
                    settlement_state = "CONFIRMED"
                except SettlementTimeoutError:
                    settlement_state = "PENDING"
                except TransactionFailedError:
                    settlement_state = "FAILED"
                    
            return SubmissionResult(
                accepted=True,
                provider_order_id=getattr(resp, "order_id", ""),
                provider_status=getattr(resp, "status", "UNKNOWN").upper(),
                provider_trade_ids=tuple(trade_ids),
                settlement_state=settlement_state,
                transaction_hashes=transaction_hashes
            )
            
        except Exception as e:
            logger.error("polymarket_submit_error", error=str(e))
            raise GatewayUnavailable(f"Transport/Network error during submit: {e}")

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")
            
        try:
            resp = await client.get_order(order_id=provider_order_id)
            if not resp:
                return SubmissionResult(
                    accepted=True,
                    provider_order_id=provider_order_id,
                    provider_status="UNKNOWN"
                )
            
            resp_dict = resp if isinstance(resp, dict) else resp.__dict__
            status = resp_dict.get("status", "UNKNOWN")
            
            return SubmissionResult(
                accepted=True,
                provider_order_id=provider_order_id,
                provider_status=status
            )
        except Exception as e:
            logger.error("polymarket_get_order_error", error=str(e))
            raise GatewayUnavailable(f"Transport/Network error during get_order: {e}")

    async def get_token_allowance(self, token_id: str) -> Decimal:
        """
        Читает allowance для конкретного conditional token.
        Проверяет два оператора: standard_exchange и neg_risk_exchange.
        При сетевой ошибке бросает GatewayUnavailable — нулевой allowance
        и невозможность его прочитать являются разными состояниями.
        """
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")
        try:
            resp = await client.get_balance_allowance(
                asset_type="CONDITIONAL", token_id=token_id
            )
        except Exception as exc:
            raise GatewayUnavailable(
                f"Cannot read conditional allowance: {exc}"
            ) from exc

        env = getattr(client, "environment", None)
        if env is None:
            raise GatewayUnavailable(
                "SDK client has no 'environment' attribute — check SDK version"
            )

        allowances: dict[str, Decimal] = {
            addr.lower(): Decimal(str(val)) / Decimal("1000000")
            for addr, val in getattr(resp, "allowances", {}).items()
        }
        required_operators = (
            str(getattr(env, "standard_exchange", "")).lower(),
            str(getattr(env, "neg_risk_exchange", "")).lower(),
        )
        return min(
            allowances.get(op, Decimal("0")) for op in required_operators
        )

    async def approve_token(self, token_id: str) -> None:
        """
        Выдаёт ERC-1155 approval для обоих операторов (standard + neg_risk).
        Вызывается ТОЛЬКО из CLI-инструмента setup_approvals.py, не из воркера.
        """
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")

        env = getattr(client, "environment", None)
        if env is None:
            raise GatewayUnavailable(
                "SDK client has no 'environment' attribute — check SDK version"
            )

        conditional_tokens = getattr(env, "conditional_tokens", None)
        if not conditional_tokens:
            raise GatewayUnavailable("Cannot resolve conditional_tokens address from environment")

        for operator_attr in ("standard_exchange", "neg_risk_exchange"):
            operator = getattr(env, operator_attr, None)
            if not operator:
                continue
            try:
                await client.approve_erc1155_for_all(
                    token_address=conditional_tokens,
                    operator_address=operator,
                    approved=True,
                )
                logger.info(
                    "approve_token_success",
                    operator=operator,
                    token_id=token_id,
                )
            except Exception as exc:
                logger.error("approve_token_error", operator=operator, error=str(exc))
                raise GatewayUnavailable(f"Failed to approve token for {operator}: {exc}") from exc

    async def fetch_order_fills(self, provider_order_id: str, token_id: str, after: str = "0") -> tuple[TradeExecution, ...]:
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")
        
        result = []
        try:
            pages = client.list_account_trades(token_id=token_id, after=after)
            async for page in pages:
                for trade in page.items:
                    maker_orders = getattr(trade, 'maker_orders', [])
                    belongs_to_order = (
                        trade.taker_order_id == provider_order_id or 
                        any(maker.order_id == provider_order_id for maker in maker_orders)
                    )
                    if not belongs_to_order:
                        continue
                        
                    if getattr(trade, "status", "CONFIRMED") != "CONFIRMED":
                        continue
                    
                    price = Decimal(str(trade.price))
                    size = Decimal(str(trade.size))
                    fee_rate = Decimal(str(getattr(trade, 'fee_rate_bps', 0)))
                    
                    fee = (price * size * fee_rate / Decimal("10000"))
                    result.append(
                        TradeExecution(
                            provider_trade_id=trade.id,
                            gateway=self.name,
                            gross_quote_usdc=price * size,
                            price=price,
                            shares=size,
                            fee_usdc=fee,
                            matched_at=trade.matched_at,
                        )
                    )
        except Exception as e:
            logger.error("failed_to_fetch_order_fills", error=str(e), order_id=provider_order_id)
            raise GatewayUnavailable(f"Failed to fetch fills: {e}")
            
        return tuple(result)

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
