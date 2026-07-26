from typing import Protocol, Tuple
from decimal import Decimal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel

class GatewayOrder(BaseModel):
    attempt_id: UUID
    market_id: str
    asset: str
    outcome_to_buy: str
    token_id: str
    side: str
    limit_price: Decimal
    requested_shares: Decimal

class ProviderFill(BaseModel):
    provider_trade_id: str
    price: Decimal
    shares: Decimal
    fee_usdc: Decimal
    matched_at: datetime

class SubmissionResult(BaseModel):
    provider_order_id: str
    status: str
    fills: Tuple[ProviderFill, ...] = ()

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
