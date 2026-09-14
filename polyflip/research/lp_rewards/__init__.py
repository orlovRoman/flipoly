"""Polymarket Liquidity Rewards Research Package (v0.1)

Empirical evaluation of LP reward economics on Polymarket.
"""

from .protocol import LPProtocol, load_protocol
from .models import MarketRewardConfig, OrderbookLevel, OrderbookSnapshot, QuotingState
from .capital_allocator import CapitalAllocator
from .scoring import calculate_sample_scores, normalize_sample_scores
from .quoting_fsm import MarketQuotingFSM

__all__ = [
    "LPProtocol",
    "load_protocol",
    "MarketRewardConfig",
    "OrderbookLevel",
    "OrderbookSnapshot",
    "QuotingState",
    "CapitalAllocator",
    "calculate_sample_scores",
    "normalize_sample_scores",
    "MarketQuotingFSM",
]
