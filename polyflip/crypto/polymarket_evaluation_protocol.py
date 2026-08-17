"""Versioned, explicit evaluation contract for LogReg Polymarket OOF replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from polyflip.crypto.logreg_polymarket_backtest import (
    LOGREG_PREDICTION_SEMANTICS,
    compute_logreg_polymarket_backtest,
)

EVALUATION_PROTOCOL_VERSION = "polymarket_logreg_eval_v1"


@dataclass(frozen=True)
class CanonicalEvaluationProtocol:
    """All parameters required to reproduce a canonical offline evaluation."""

    protocol_version: str = EVALUATION_PROTOCOL_VERSION
    strategy_branch: str = "COMBINED"
    min_edge: float = 0.03
    fee_rate: float = 0.02
    slippage_pct: float = 0.0
    cost_buffer: float = 0.0
    stake_usdc: float = 1.0
    min_price: float = 0.05
    max_price: float = 0.95
    outsider_max_price: float = 0.45

    def __post_init__(self) -> None:
        if self.strategy_branch != "COMBINED":
            raise ValueError("canonical protocol requires strategy_branch=COMBINED")
        if self.slippage_pct < 0.0 or self.slippage_pct >= 1.0:
            raise ValueError("slippage_pct must be an explicit number in [0, 1)")
        if self.cost_buffer < 0.0 or self.fee_rate < 0.0 or self.min_edge < 0.0:
            raise ValueError("cost and edge parameters must be non-negative")
        if self.stake_usdc <= 0.0:
            raise ValueError("stake_usdc must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CANONICAL_EVALUATION_PROTOCOL = CanonicalEvaluationProtocol()


def evaluate_logreg_with_protocol(
    frame: pd.DataFrame,
    p_flip: Any,
    quotes: pd.DataFrame | None,
    *,
    protocol: CanonicalEvaluationProtocol = CANONICAL_EVALUATION_PROTOCOL,
) -> dict[str, Any]:
    """Replay LogReg OOF probabilities with the immutable protocol contract."""
    result = compute_logreg_polymarket_backtest(
        frame,
        p_flip,
        quotes,
        strategy_branch=protocol.strategy_branch,
        min_edge=protocol.min_edge,
        fee_rate=protocol.fee_rate,
        slippage_pct=protocol.slippage_pct,
        cost_buffer=protocol.cost_buffer,
        stake_usdc=protocol.stake_usdc,
        min_price=protocol.min_price,
        max_price=protocol.max_price,
        outsider_max_price=protocol.outsider_max_price,
    )
    result["evaluation_metadata"] = {
        "protocol": protocol.as_dict(),
        "evaluator": "compute_logreg_polymarket_backtest",
        "metrics_schema_version": result.get("metrics_schema_version"),
        "prediction_semantics": LOGREG_PREDICTION_SEMANTICS,
        "probability_conversion": (
            "p_flip -> p_yes: 1-p_flip when YES is favourite (mid_price > 0.5); "
            "otherwise p_yes=p_flip"
        ),
        "selection_rule": (
            "COMBINED evaluates BUY_YES and BUY_NO candidates and executes at most "
            "one selected side per market"
        ),
        "entry_snapshot_fields": [
            "market_id",
            "recorded_at",
            "yes_price",
            "no_price",
            "best_bid",
            "best_ask",
            "mid_price",
            "final_outcome",
        ],
        "trade_evidence_fields": [
            "market_id",
            "side",
            "price",
            "p_win",
            "entry_time",
            "outcome",
            "pnl",
        ],
    }
    return result
