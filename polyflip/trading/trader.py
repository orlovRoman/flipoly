import os
import structlog
import time
import asyncio
from typing import Optional, Dict, Any, Literal
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON
from polyflip.trading.schemas import BalanceResult, BalanceStatus, TradeExecution, ExecutionFees, ExecutionStatus

logger = structlog.get_logger(__name__)

class PolyTrader:
    def __init__(self):
        self.host = "https://clob.polymarket.com"
        self.chain_id = POLYGON
        # Кеширование клиента для BUG-N05
        self._client_cache: Optional[ClobClient] = None
        self._client_cache_time: float = 0
        self._last_key_hash: str = ""
        self._last_address: str = ""

    def get_client(self) -> Optional[ClobClient]:
        # BUG-T01 FIX: Читаем ключ перед каждой сделкой
        private_key = os.getenv("POLYGON_PRIVATE_KEY")
        address = os.getenv("POLYGON_ADDRESS")
        
        if not private_key or not address:
            return None
            
        import hashlib
        import hmac
        
        now = time.time()
        current_key_hash = hashlib.sha256(private_key.encode()).hexdigest()
        
        # Если прошло меньше 5 минут и ключи не изменились — используем кэш
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
            
            # Сохраняем в кэш
            self._client_cache = client
            self._client_cache_time = now
            self._last_key_hash = current_key_hash
            self._last_address = address
            
            return client
        except Exception as e:
            logger.error("failed_to_init_clob_client", error=str(e))
            return None

    async def execute_trade(
        self, 
        market_id: str, 
        token_id: str, 
        side: Literal["BUY", "SELL"],
        price: float, 
        size: float
    ) -> TradeExecution:
        """Отправляет лимитный ордер (FOK) и возвращает TradeExecution."""
        logger.info("executing_trade", market_id=market_id, side=side, price=price, size=size)
        
        attempt_id = uuid4()
        now = datetime.now(timezone.utc)
        
        fees = ExecutionFees(
            platform_fee_usdc=Decimal("0"),
            builder_fee_usdc=Decimal("0"),
            network_fee_native=None,
            network_fee_symbol=None,
            network_fee_usdc=None,
            fee_source="ESTIMATED"
        )
        
        if self.get_client() is None:
            logger.info("paper_trade_executed", market_id=market_id, side=side, price=price, size=size)
            return TradeExecution(
                attempt_id=attempt_id,
                provider_order_id="PAPER_" + str(attempt_id),
                provider_status="PAPER",
                status="PAPER_FILLED",
                side=side,
                order_type="PAPER",
                token_id=token_id,
                original_requested_shares=Decimal(str(size)),
                submitted_shares=Decimal(str(size)),
                filled_shares=Decimal(str(size)),
                net_position_delta_shares=Decimal(str(size)) if side == "BUY" else -Decimal(str(size)),
                average_price=Decimal(str(price)),
                gross_quote_usdc=Decimal(str(size * price)),
                net_quote_usdc=Decimal(str(size * price)),
                liquidity_role="UNKNOWN",
                fees=fees,
                trade_ids=tuple(),
                transaction_hashes=tuple(),
                submitted_at=now,
                observed_at=now,
                error_code=None,
                error_message=None
            )
            
        client = self.get_client()
        if not client:
            return TradeExecution(
                attempt_id=attempt_id,
                provider_order_id=None,
                provider_status=None,
                status="REJECTED",
                side=side,
                order_type="FOK",
                token_id=token_id,
                original_requested_shares=Decimal(str(size)),
                submitted_shares=Decimal("0"),
                filled_shares=Decimal("0"),
                net_position_delta_shares=Decimal("0"),
                average_price=None,
                gross_quote_usdc=None,
                net_quote_usdc=None,
                liquidity_role="UNKNOWN",
                fees=fees,
                trade_ids=tuple(),
                transaction_hashes=tuple(),
                submitted_at=now,
                observed_at=now,
                error_code="CLIENT_INIT_FAILED",
                error_message="Polymarket client not initialized"
            )
            
        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id
            )
            
            resp = client.create_and_post_order(order_args, order_type=OrderType.FOK)
            observed = datetime.now(timezone.utc)
            
            if resp and resp.get("success"):
                order_id = resp.get("orderID")
                logger.info("trade_success", order_id=order_id, size=size)
                # Возвращаем UNKNOWN, так как ордер только ПРИНЯТ
                return TradeExecution(
                    attempt_id=attempt_id,
                    provider_order_id=order_id,
                    provider_status="LIVE",
                    status="UNKNOWN",
                    side=side,
                    order_type="FOK",
                    token_id=token_id,
                    original_requested_shares=Decimal(str(size)),
                    submitted_shares=Decimal(str(size)),
                    filled_shares=Decimal("0"),
                    net_position_delta_shares=None,
                    average_price=None,
                    gross_quote_usdc=None,
                    net_quote_usdc=None,
                    liquidity_role="UNKNOWN",
                    fees=fees,
                    trade_ids=tuple(),
                    transaction_hashes=tuple(),
                    submitted_at=now,
                    observed_at=observed,
                    error_code=None,
                    error_message=None
                )
            
            err = resp.get("errorMsg") if resp else "Unknown API error"
            logger.warning("trade_rejected", error=err, size=size)
            return TradeExecution(
                attempt_id=attempt_id,
                provider_order_id=None,
                provider_status=None,
                status="REJECTED",
                side=side,
                order_type="FOK",
                token_id=token_id,
                original_requested_shares=Decimal(str(size)),
                submitted_shares=Decimal(str(size)),
                filled_shares=Decimal("0"),
                net_position_delta_shares=Decimal("0"),
                average_price=None,
                gross_quote_usdc=None,
                net_quote_usdc=None,
                liquidity_role="UNKNOWN",
                fees=fees,
                trade_ids=tuple(),
                transaction_hashes=tuple(),
                submitted_at=now,
                observed_at=observed,
                error_code="REJECTED",
                error_message=err
            )
                    
        except Exception as e:
            logger.exception("trade_exception", error=str(e))
            return TradeExecution(
                attempt_id=attempt_id,
                provider_order_id=None,
                provider_status=None,
                status="UNKNOWN",
                side=side,
                order_type="FOK",
                token_id=token_id,
                original_requested_shares=Decimal(str(size)),
                submitted_shares=Decimal(str(size)),
                filled_shares=Decimal("0"),
                net_position_delta_shares=None,
                average_price=None,
                gross_quote_usdc=None,
                net_quote_usdc=None,
                liquidity_role="UNKNOWN",
                fees=fees,
                trade_ids=tuple(),
                transaction_hashes=tuple(),
                submitted_at=now,
                observed_at=datetime.now(timezone.utc),
                error_code="NETWORK_EXCEPTION",
                error_message=str(e)
            )

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Получает статус ордера по ID."""
        client = self.get_client()
        if not client:
            return None
        try:
            return client.get_order(order_id)
        except Exception as e:
            logger.error("get_order_exception", order_id=order_id, error=str(e))
            return None

    async def get_balance(self, token_id: str) -> BalanceResult:
        """Получает текущий баланс токена."""
        if self.get_client() is None:
            return BalanceResult(
                status="PAPER",
                wallet_address="PAPER_WALLET",
                token_id=token_id,
                available_shares=Decimal("0"),
                total_shares=Decimal("0"),
                locked_shares=Decimal("0"),
                observed_at=datetime.now(timezone.utc),
                source="PAPER_TRADER"
            )

        client = self.get_client()
        if not client:
            return BalanceResult(
                status="TRANSPORT_ERROR",
                wallet_address="UNKNOWN",
                token_id=token_id,
                available_shares=None,
                total_shares=None,
                locked_shares=None,
                observed_at=datetime.now(timezone.utc),
                source="PY_CLOB_CLIENT",
                error_message="Client not initialized"
            )
            
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams
            resp = client.get_balance_allowance(BalanceAllowanceParams(asset_type="conditional"))
            
            wallet_addr = getattr(client, 'address', "UNKNOWN")
            now = datetime.now(timezone.utc)

            if isinstance(resp, list):
                for b in resp:
                    if b.get("asset_id") == token_id or b.get("token_id") == token_id:
                        balance_val = Decimal(str(b.get("balance", 0)))
                        return BalanceResult(
                            status="OK",
                            wallet_address=wallet_addr,
                            token_id=token_id,
                            available_shares=balance_val,
                            total_shares=balance_val,
                            locked_shares=Decimal("0"),
                            observed_at=now,
                            source="PY_CLOB_CLIENT"
                        )
                return BalanceResult(
                    status="TOKEN_NOT_FOUND",
                    wallet_address=wallet_addr,
                    token_id=token_id,
                    available_shares=None,
                    total_shares=None,
                    locked_shares=None,
                    observed_at=now,
                    source="PY_CLOB_CLIENT",
                    error_message=f"Token {token_id} not found in balance list"
                )
            
            return BalanceResult(
                status="PARSE_ERROR",
                wallet_address=wallet_addr,
                token_id=token_id,
                available_shares=None,
                total_shares=None,
                locked_shares=None,
                observed_at=now,
                source="PY_CLOB_CLIENT",
                error_message=f"Unexpected response format: {type(resp)}"
            )

        except Exception as e:
            logger.error("get_balance_exception", token_id=token_id, error=str(e))
            return BalanceResult(
                status="TRANSPORT_ERROR",
                wallet_address="UNKNOWN",
                token_id=token_id,
                available_shares=None,
                total_shares=None,
                locked_shares=None,
                observed_at=datetime.now(timezone.utc),
                source="PY_CLOB_CLIENT",
                error_message=str(e)
            )
