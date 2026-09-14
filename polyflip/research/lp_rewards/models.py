from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QuotingState(str, Enum):
    FLAT = "FLAT"
    QUOTING_BOTH = "QUOTING_BOTH"
    LONG_YES = "LONG_YES"
    LONG_NO = "LONG_NO"
    COMPLETE_SET = "COMPLETE_SET"
    MERGING = "MERGING"
    EXITING = "EXITING"
    HALTED = "HALTED"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderbookLevel(BaseModel):
    price: Decimal
    size: Decimal
    order_age_sec: Decimal = Decimal("0.0")


class OrderbookSnapshot(BaseModel):
    condition_id: str
    asset_id: str
    bids: List[OrderbookLevel] = Field(default_factory=list)
    asks: List[OrderbookLevel] = Field(default_factory=list)
    timestamp_ns: int
    valid_from_ns: int
    valid_to_ns: Optional[int] = None
    is_uncertain: bool = False


class MarketRewardConfig(BaseModel):
    condition_id: str
    question: str
    rewards_daily_rate: Decimal
    rewards_max_spread: Decimal
    rewards_min_size: Decimal
    oas: Decimal = Decimal("5.0")  # Order age seconds from CLOB info
    neg_risk: bool = False
    yes_token_id: str
    no_token_id: str
    end_date_iso: Optional[str] = None
    taker_fee_rate: Decimal = Decimal("0.00")  # taker fee from feeSchedule (usually 0.0 or e.g. 0.001)


class VirtualOrder(BaseModel):
    order_id: str
    condition_id: str
    asset_id: str
    side: OrderSide
    price: Decimal
    size: Decimal
    placed_at_ns: int
    post_only: bool = True
    time_in_force: str = "GTD"
    filled_size: Decimal = Decimal("0.0")
    status: str = "OPEN"  # OPEN, PARTIALLY_FILLED, FILLED, CANCELLED


class VirtualFill(BaseModel):
    fill_id: str
    order_id: str
    condition_id: str
    asset_id: str
    side: OrderSide
    price: Decimal
    size: Decimal
    timestamp_ns: int
    queue_depletion_ratio: Decimal
    taker_fee_paid: Decimal = Decimal("0.0")
    markout_5s: Optional[Decimal] = None
    markout_60s: Optional[Decimal] = None
    markout_15m: Optional[Decimal] = None


class MarketPosition(BaseModel):
    condition_id: str
    yes_inventory: Decimal = Decimal("0.0")
    no_inventory: Decimal = Decimal("0.0")
    cash_invested: Decimal = Decimal("0.0")
    state: QuotingState = QuotingState.FLAT
    state_entered_at_ns: int = 0
    yes_fill_cost: Decimal = Decimal("0.0")
    no_fill_cost: Decimal = Decimal("0.0")
    complete_sets_merged: Decimal = Decimal("0.0")
    realized_trading_pnl: Decimal = Decimal("0.0")
    taker_fees_paid: Decimal = Decimal("0.0")


class SampleScore(BaseModel):
    condition_id: str
    timestamp_ns: int
    p_mid_star: Optional[Decimal] = None
    q_one: Decimal = Decimal("0.0")
    q_two: Decimal = Decimal("0.0")
    q_min: Decimal = Decimal("0.0")
    status: str = "VALID"  # VALID, MID_UNCERTAIN, BOOK_UNCERTAIN
