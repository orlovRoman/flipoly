from typing import Protocol
from decimal import Decimal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from polyflip.execution.gateways.exceptions import (
    GatewayError,
    GatewayUnavailable,
    GatewayOrderRejected,
    GatewaySubmissionUnknown,
)

__all__ = [
    "BalanceResult",
    "GatewayReadiness",
    "GatewayUnavailable",
    "GatewayError",
    "GatewayOrderRejected",
    "GatewaySubmissionUnknown",
    "GatewayOrder",
    "ProviderFill",
    "ProviderOrderState",
    "TradeExecution",
    "ExecutionGateway",
]


class BalanceResult(BaseModel):
    balance_usdc: Decimal
    collateral_allowances: dict[str, Decimal] = Field(default_factory=dict)
    conditional_allowances_checked: int = 0
    conditional_allowance_ready: bool | None = None
    checked_at: datetime
    raw_asset_type: str = "COLLATERAL"


class GatewayReadiness(BaseModel):
    ready: bool
    gateway: str
    wallet_address: str | None
    balance: BalanceResult | None
    credentials_loaded: bool
    client_initialized: bool
    collateral_allowance_ready: bool
    conditional_allowance_ready: bool | None
    network_chain_id: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    checked_at: datetime


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
    # Upper execution-price bound used when FAK_RETRY refreshes the quote.
    max_acceptable_price: Decimal | None = None
    # Unix timestamp used by native GTD orders.  It is intentionally optional
    # so existing FAK/GTC callers keep their current contract.
    expiration: int | None = None
    # Reject an order that would immediately take resting liquidity.
    post_only: bool = False


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
    transaction_hash: str | None = None


class SubmissionResult(BaseModel):
    accepted: bool
    provider_order_id: str | None = None
    provider_status: str
    provider_trade_ids: tuple[str, ...] = ()
    rejection_code: str | None = None
    error_message: str | None = None
    settlement_state: str = "PENDING"
    transaction_hashes: tuple[str, ...] = ()
    # Заполняется шлюзами, которые возвращают fills синхронно (SHADOW, FAKE).
    # Если непустое — worker пропускает fetch_order_fills.
    fills: tuple["TradeExecution", ...] = ()
    # Maker execution telemetry. These fields describe an optional local
    # reprice attempt without changing the provider status contract.
    submitted_limit_price: Decimal | None = None
    submitted_requested_shares: Decimal | None = None
    maker_status: str | None = None
    maker_attempts: int = 1
    maker_best_bid: Decimal | None = None
    maker_best_ask: Decimal | None = None
    paper_quote_price: Decimal | None = None
    paper_available_shares: Decimal | None = None
    paper_delay_seconds: float | None = None
    paper_slippage_usdc: Decimal | None = None
    paper_fee_usdc: Decimal | None = None



class ExecutionGateway(Protocol):
    name: str

    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        """Submit an order to the execution venue."""
        ...

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        """Fetch the latest status for an order."""
        ...

    async def fetch_order_fills(
        self, provider_order_id: str, token_id: str, after: str = "0"
    ) -> tuple[TradeExecution, ...]:
        """Fetch the actual trade fills for an order."""
        ...

    async def get_token_allowance(self, token_id: str) -> Decimal:
        """Get the allowance for a specific conditional token."""
        ...

    async def approve_token(self, token_id: str) -> None:
        """Approve a specific conditional token."""
        ...

    async def get_readiness(
        self,
        conditional_token_ids: tuple[str, ...] = (),
    ) -> GatewayReadiness:
        """Get the current readiness status of the gateway."""
        ...
