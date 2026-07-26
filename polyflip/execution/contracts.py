from typing import Protocol, Tuple
from decimal import Decimal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel

class GatewayUnavailable(Exception):
    pass

class GatewayOrder(BaseModel):
    attempt_id: UUID
    market_id: str
    asset: str
    outcome_to_buy: str
    token_id: str
    side: str
    limit_price: Decimal
    requested_shares: Decimal
    max_spend_usdc: Decimal | None = None

class ProviderFill(BaseModel):
    provider_trade_id: str
    price: Decimal
    shares: Decimal
    fee_usdc: Decimal
    matched_at: datetime

class ProviderOrderState(BaseModel):
    provider_order_id: str
    status: str
    filled_shares: Decimal = Decimal("0")
    filled_usdc: Decimal = Decimal("0")
    remaining_shares: Decimal = Decimal("0")
    fee_usdc: Decimal = Decimal("0")
    created_at: datetime
    updated_at: datetime

class TradeExecution(BaseModel):
    provider_trade_id: str
    gateway: str
    gross_quote_usdc: Decimal
    price: Decimal
    shares: Decimal
    fee_usdc: Decimal
    matched_at: datetime

class SubmissionResult(BaseModel):
    provider_order_id: str
    status: str
    state: ProviderOrderState | None = None
    fills: Tuple[TradeExecution, ...] = ()

class ExecutionGateway(Protocol):
    name: str

    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        """Submit an order to the execution venue."""
        ...

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        """Fetch the latest status and fills for an order."""
        ...

    async def get_balance(self) -> Decimal:
        """Get the current USDC balance available for trading."""
        ...
