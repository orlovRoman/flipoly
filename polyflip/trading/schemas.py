from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

ExecutionStatus = Literal[
    "SUBMITTED",
    "LIVE",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCELLED",
    "REJECTED",
    "UNKNOWN",
    "PAPER_FILLED",
]

@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionFees:
    platform_fee_usdc: Decimal | None
    builder_fee_usdc: Decimal | None
    # Фактические затраты пользователя на сеть
    network_fee_native: Decimal | None
    network_fee_symbol: str | None
    network_fee_usdc: Decimal | None
    fee_source: Literal[
        "REPORTED",
        "CALCULATED",
        "ESTIMATED",
        "CONFIRMED_ZERO",
        "UNKNOWN",
    ]

@dataclass(frozen=True, slots=True, kw_only=True)
class TradeExecution:
    attempt_id: UUID
    provider_order_id: str | None
    provider_status: str | None
    status: ExecutionStatus
    side: Literal["BUY", "SELL"]
    order_type: Literal["FOK", "FAK", "GTC", "GTD", "PAPER"]
    token_id: str
    original_requested_shares: Decimal
    submitted_shares: Decimal
    filled_shares: Decimal
    # Реальное изменение токенов после подтверждения баланса
    net_position_delta_shares: Decimal | None
    average_price: Decimal | None
    gross_quote_usdc: Decimal | None
    net_quote_usdc: Decimal | None
    liquidity_role: Literal["MAKER", "TAKER", "MIXED", "UNKNOWN"]
    fees: ExecutionFees
    trade_ids: tuple[str, ...]
    transaction_hashes: tuple[str, ...]
    submitted_at: datetime
    observed_at: datetime
    error_code: str | None = None
    error_message: str | None = None

BalanceStatus = Literal[
    "OK",
    "TRANSPORT_ERROR",
    "PARSE_ERROR",
    "TOKEN_NOT_FOUND",
    "STALE",
    "PAPER",
]

@dataclass(frozen=True, slots=True, kw_only=True)
class BalanceResult:
    status: BalanceStatus
    wallet_address: str
    token_id: str
    available_shares: Decimal | None
    total_shares: Decimal | None
    locked_shares: Decimal | None
    observed_at: datetime | None
    source: str
    block_number: int | None = None
    error_code: str | None = None
    error_message: str | None = None
