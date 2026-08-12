from dataclasses import replace
from polymarket.environments import PRODUCTION
import structlog
from typing import Optional
from decimal import Decimal
from datetime import datetime, timezone
import asyncio

from polymarket import AsyncSecureClient, RelayerApiKey
from eth_account import Account
from polymarket.errors import (
    TimeoutError as SettlementTimeoutError,
    TransactionFailedError,
)
from polyflip.execution.contracts import (
    GatewayOrder,
    TradeExecution,
    SubmissionResult,
    GatewayUnavailable,
    GatewayReadiness,
    BalanceResult,
)
from polyflip.execution.gateways.exceptions import (
    GatewayOrderRejected,
    GatewaySubmissionUnknown,
)

logger = structlog.get_logger(__name__)


def validate_signer(private_key: str, expected_address: str) -> str:
    try:
        actual_address = Account.from_key(private_key).address
    except Exception as e:
        raise GatewayUnavailable(f"Invalid POLYGON_PRIVATE_KEY format: {e}")

    if actual_address.lower() != expected_address.lower():
        raise GatewayUnavailable(
            "POLYGON_PRIVATE_KEY does not match POLYMARKET_RELAYER_API_KEY_ADDRESS"
        )

    return actual_address


class PolymarketExecutionGateway:
    name = "POLYMARKET"

    def __init__(
        self,
        private_key: str,
        wallet_address: str,
        relayer_api_key: str,
        relayer_api_key_address: str,
        host: str = "https://clob.polymarket.com",
    ):
        self._private_key = private_key
        self._wallet_address = wallet_address
        self._relayer_api_key = relayer_api_key
        self._relayer_api_key_address = relayer_api_key_address
        self._host = host
        self._environment = replace(
            PRODUCTION,
            clob_url=host.rstrip("/"),
        )
        self._client_cache: Optional[AsyncSecureClient] = None
        self._client_lock = asyncio.Lock()

    @property
    def credentials_loaded(self) -> bool:
        return all(
            (
                self._private_key,
                self._wallet_address,
                self._relayer_api_key,
                self._relayer_api_key_address,
            )
        )

    def _validate_credentials(self) -> None:
        if not self.credentials_loaded:
            raise GatewayUnavailable("Missing Polymarket credentials")

        validate_signer(self._private_key, self._relayer_api_key_address)

    async def get_client(self) -> Optional[AsyncSecureClient]:
        if not self.credentials_loaded:
            logger.error("missing_polygon_credentials")
            return None

        self._validate_credentials()

        async with self._client_lock:
            if self._client_cache:
                return self._client_cache

            try:
                client = await AsyncSecureClient.create(
                    private_key=self._private_key,
                    wallet=self._wallet_address,
                    api_key=RelayerApiKey(
                        key=self._relayer_api_key,
                        address=self._relayer_api_key_address,
                    ),
                    environment=self._environment,
                )
                self._client_cache = client
                return client
            except Exception as e:
                self._client_cache = None
                logger.error("failed_to_init_async_secure_client", error=str(e))
                raise GatewayUnavailable(f"Polymarket client init failed: {e}")

    async def invalidate_client(self) -> None:
        """Сбрасывает кэшированный клиент при сетевых/SSL ошибках."""
        import inspect

        async with self._client_lock:
            client = self._client_cache
            self._client_cache = None

        if client is None:
            return

        try:
            close_fn = getattr(client, "close", None)
            if close_fn:
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            logger.warning(
                "polymarket_client_close_failed",
                error=str(exc),
            )

    async def submit(
        self, order: GatewayOrder, order_type: str = "FAK"
    ) -> SubmissionResult:
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")

        try:
            normalized_order_type = order_type.upper()
            side = order.side.upper()

            if normalized_order_type in {"GTC", "GTD", "GTC_TTL"}:
                if order.requested_shares <= 0:
                    raise GatewayOrderRejected(
                        "Limit order requires requested_shares > 0"
                    )

                # polymarket-client 0.2.0 represents GTC/GTD through
                # place_limit_order: no expiration means GTC, while a future
                # expiration means GTD. The external GTC_TTL strategy still
                # cancels this resting order after LIVE_GTC_TTL_SECONDS.
                expiration = None
                if normalized_order_type == "GTD":
                    # A TP order supplies the market end as its native GTD
                    # expiry.  Keep a safe fallback for older callers, and
                    # fail closed when the exchange's three-minute minimum
                    # cannot be met rather than placing an order that outlives
                    # the market.
                    now_ts = int(datetime.now(timezone.utc).timestamp())
                    requested_expiration = order.expiration
                    expiration = requested_expiration or (now_ts + 300)
                    if expiration <= now_ts + 180:
                        raise GatewayOrderRejected(
                            "GTD_EXPIRATION_TOO_SOON: expiration must be at least "
                            "180 seconds in the future"
                        )

                resp = await client.place_limit_order(
                    token_id=order.token_id,
                    price=str(order.limit_price),
                    size=str(order.requested_shares),
                    side=side,
                    post_only=False,
                    expiration=expiration,
                )
            elif normalized_order_type in {"FAK", "FOK"}:
                if side == "BUY":
                    amount_limit = (
                        str(order.max_spend_usdc) if order.max_spend_usdc else "0"
                    )
                    resp = await client.place_market_order(
                        token_id=order.token_id,
                        side="BUY",
                        amount=amount_limit,
                        max_spend=amount_limit,
                        max_price=str(order.limit_price),
                        order_type=normalized_order_type,
                    )
                else:
                    resp = await client.place_market_order(
                        token_id=order.token_id,
                        side="SELL",
                        shares=str(order.requested_shares),
                        min_price=str(order.limit_price),
                        order_type=normalized_order_type,
                    )
            else:
                raise GatewayOrderRejected(
                    f"Unsupported Polymarket order type: {order_type}"
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
                transaction_hashes=transaction_hashes,
            )

        except (GatewayOrderRejected, GatewaySubmissionUnknown, GatewayUnavailable):
            raise
        except Exception as e:
            err_msg = str(e)
            logger.error("polymarket_submit_error", error=err_msg)
            err_lower = err_msg.lower()

            rejection_keywords = [
                "invalid amount",
                "min size",
                "minimum order",
                "validation error",
                "insufficient funds",
                "invalid price",
                "order size too small",
                "bad request",
                "below minimum",
            ]
            network_keywords = [
                "connectionterminated",
                "timeout",
                "connecterror",
                "connection reset",
            ]
            fak_no_liquidity_markers = (
                "no orders found to match with fak order",
                "there are no matching orders",
            )

            if any(marker in err_lower for marker in fak_no_liquidity_markers):
                mode_label = (
                    "GTD"
                    if order_type.upper() in ("GTC", "GTD", "GTC_TTL")
                    else order_type.upper()
                )
                raise GatewayOrderRejected(
                    f"NO_LIQUIDITY_{mode_label}: [{mode_label}] "
                    "Заявка не нашла встречной ликвидности в стакане"
                ) from e

            if any(keyword in err_lower for keyword in rejection_keywords):
                raise GatewayOrderRejected(f"Order rejected by Polymarket: {e}") from e

            if any(keyword in err_lower for keyword in network_keywords):
                raise GatewaySubmissionUnknown(
                    f"Submission unknown due to network error: {e}"
                ) from e

            raise GatewayUnavailable(
                f"Unexpected Polymarket submission error: {e}"
            ) from e

    async def cancel_order(self, provider_order_id: str) -> bool:
        client = await self.get_client()
        if not client:
            return False
        try:
            if hasattr(client, "cancel_order"):
                await client.cancel_order(order_id=provider_order_id)
            elif hasattr(client, "cancel"):
                await client.cancel(order_id=provider_order_id)
            else:
                logger.error(
                    "cancel_order_not_supported_by_client",
                    provider_order_id=provider_order_id,
                )
                return False
            return True
        except Exception as e:
            logger.warning(
                "cancel_order_failed", provider_order_id=provider_order_id, error=str(e)
            )
            return False

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
                    provider_status="UNKNOWN",
                )

            resp_dict = resp if isinstance(resp, dict) else resp.__dict__
            status = resp_dict.get("status", "UNKNOWN")

            return SubmissionResult(
                accepted=True,
                provider_order_id=provider_order_id,
                provider_status=status,
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
        return min(allowances.get(op, Decimal("0")) for op in required_operators)

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
            raise GatewayUnavailable(
                "Cannot resolve conditional_tokens address from environment"
            )

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
                raise GatewayUnavailable(
                    f"Failed to approve token for {operator}: {exc}"
                ) from exc

    async def fetch_order_fills(
        self, provider_order_id: str, token_id: str, after: str = "0"
    ) -> tuple[TradeExecution, ...]:
        client = await self.get_client()
        if not client:
            raise GatewayUnavailable("Polymarket client not initialized")

        result = []
        try:
            pages = client.list_account_trades(token_id=token_id, after=after)
            async for page in pages:
                for trade in page.items:
                    maker_orders = getattr(trade, "maker_orders", [])
                    belongs_to_order = trade.taker_order_id == provider_order_id or any(
                        maker.order_id == provider_order_id for maker in maker_orders
                    )
                    if not belongs_to_order:
                        continue

                    if getattr(trade, "status", "CONFIRMED") != "CONFIRMED":
                        continue

                    price = Decimal(str(trade.price))
                    size = Decimal(str(trade.size))
                    fee_rate = Decimal(str(getattr(trade, "fee_rate_bps", 0)))

                    fee = price * size * fee_rate / Decimal("10000")
                    result.append(
                        TradeExecution(
                            provider_trade_id=trade.id,
                            gateway=self.name,
                            gross_quote_usdc=price * size,
                            price=price,
                            shares=size,
                            fee_usdc=fee,
                            matched_at=trade.matched_at,
                            transaction_hash=getattr(trade, "transaction_hash", None),
                        )
                    )
        except Exception as e:
            logger.error(
                "failed_to_fetch_order_fills", error=str(e), order_id=provider_order_id
            )
            raise GatewayUnavailable(f"Failed to fetch fills: {e}")

        return tuple(result)

    async def get_readiness(
        self, conditional_token_ids: tuple[str, ...] = ()
    ) -> GatewayReadiness:
        credentials_loaded = self.credentials_loaded

        readiness = GatewayReadiness(
            ready=False,
            gateway=self.name,
            wallet_address=self._wallet_address,
            balance=None,
            credentials_loaded=credentials_loaded,
            client_initialized=False,
            collateral_allowance_ready=False,
            conditional_allowance_ready=None,
            network_chain_id=getattr(
                self._environment,
                "chain_id",
                None,
            ),
            checked_at=datetime.now(timezone.utc),
        )

        try:
            client = await self.get_client()
        except GatewayUnavailable as exc:
            readiness.error_message = str(exc)
            return readiness

        if client is None:
            readiness.error_message = "Polymarket client initialization failed"
            return readiness

        readiness.client_initialized = True

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

            token_allowances = [
                await self.get_token_allowance(token_id)
                for token_id in conditional_token_ids
            ]

            conditional_ready: bool | None = None
            if conditional_token_ids:
                conditional_ready = all(value > 0 for value in token_allowances)

            balance_result = BalanceResult(
                raw_balance_usdc=raw_balance,
                balance_usdc=balance_usdc,
                collateral_allowances=parsed_allowances,
                collateral_allowance_ready=collateral_ready,
                conditional_allowances_checked=len(conditional_token_ids),
                conditional_allowance_ready=conditional_ready,
                checked_at=datetime.now(timezone.utc),
                raw_asset_type="COLLATERAL",
            )

            readiness.balance = balance_result
            readiness.collateral_allowance_ready = collateral_ready
            readiness.conditional_allowance_ready = conditional_ready

            readiness.ready = (
                collateral_ready
                and (conditional_ready is not False)
                and (balance_usdc >= Decimal("5.00"))
            )
            return readiness

        except Exception as e:
            logger.error("polymarket_gateway_readiness_error", error=str(e))
            readiness.error_message = str(e)
            return readiness
