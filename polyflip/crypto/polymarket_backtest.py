"""Polymarket OOF backtesting helpers for canonical LightGBM models.

The LightGBM candle backtester historically measured the next Binance return.
That is useful as a diagnostic, but it is not the trading PnL of a binary
Polymarket token.  This module keeps the accounting separate and evaluates
OOF probabilities against the actual YES/NO quote and the canonical market
resolution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import MarketSnapshot
from polyflip.trading.weighted_policy import (
    WeightedPolicyConfig,
    market_yes_probability,
    WeightedSideQuote,
    select_weighted_side,
)


DEFAULT_MIN_EDGE = 0.04
DEFAULT_COST_BUFFER = 0.02
DEFAULT_FEE_RATE = 0.002
DEFAULT_MIN_PRICE = 0.05
DEFAULT_MAX_PRICE = 0.95
DEFAULT_OUTSIDER_MAX_PRICE = 0.45
DEFAULT_STAKE_USDC = 1.0
DEFAULT_WEIGHTED_FEE_RATE = 0.07
DEFAULT_WEIGHTED_FEE_EXPONENT = 1.0
DEFAULT_WEIGHTED_SLIPPAGE_RATE = 0.005


@dataclass(frozen=True)
class Candidate:
    side: str
    price: float
    p_win: float
    gross_edge: float
    net_edge: float


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _quote_prices(row: pd.Series) -> tuple[float | None, float | None]:
    """Return executable YES and NO asks from a market snapshot.

    A NO ask is ``1 - YES bid``.  Falling back to the midpoint is explicitly
    documented and only used when the snapshot did not persist top-of-book.
    """
    mid = _as_float(row.get("mid_price"))
    spread = max(_as_float(row.get("spread"), 0.0) or 0.0, 0.0)
    yes_ask = _as_float(row.get("best_ask"))
    yes_bid = _as_float(row.get("best_bid"))
    if yes_ask is None and mid is not None:
        yes_ask = mid + spread / 2.0
    if yes_bid is None and mid is not None:
        yes_bid = mid - spread / 2.0
    if yes_ask is None or yes_bid is None:
        return None, None
    return (
        float(np.clip(yes_ask, 0.001, 0.999)),
        float(np.clip(1.0 - yes_bid, 0.001, 0.999)),
    )


def _market_mid_price(
    row: pd.Series,
    yes_ask: float,
    no_ask: float,
) -> float:
    """Return the YES mid used as the market prior in weighted mode."""
    mid = _as_float(row.get("mid_price"))
    if mid is None:
        yes_bid = _as_float(row.get("best_bid"))
        if yes_bid is not None:
            mid = (yes_ask + yes_bid) / 2.0
        else:
            # ``no_ask = 1 - YES bid`` when the quote is top-of-book.  This
            # fallback is only for old artifacts that persisted asks but not
            # a midpoint or bid.
            mid = (yes_ask + (1.0 - no_ask)) / 2.0
    normalized = market_yes_probability(
        yes_ask=yes_ask, no_ask=no_ask, fallback_yes=mid,
    )
    return float(np.clip(normalized if normalized is not None else 0.5, 0.001, 0.999))


def _normalize_score_series(
    values: Iterable[float] | float | None,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    """Normalize an optional scalar/iterable model output for OOF replay."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric iterable")
    try:
        if np.isscalar(values):
            array = np.full(expected_length, float(values), dtype=float)
        else:
            array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if len(array) != expected_length:
        raise ValueError(f"{name} must have the same length as frame")
    return array


def _price_bucket(price: float) -> str:
    if price < 0.20:
        return "0.05-0.20"
    if price < 0.35:
        return "0.20-0.35"
    if price < 0.50:
        return "0.35-0.50"
    if price < 0.65:
        return "0.50-0.65"
    if price < 0.80:
        return "0.65-0.80"
    return "0.80-0.95"


def _phase_bucket(time_left_min: float | None, regime: str | None) -> str:
    if regime:
        return str(regime)
    left = _as_float(time_left_min, 0.0) or 0.0
    if left <= 5.0:
        return "FINAL_0_5"
    if left <= 10.0:
        return "MIDDLE_5_10"
    return "EARLY_10_15"


def _candidate(
    side: str,
    price: float,
    p_win: float,
    *,
    cost_buffer: float,
) -> Candidate:
    gross = float(p_win - price)
    return Candidate(side, price, float(p_win), gross, gross - cost_buffer)


def _select_candidate(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
    *,
    strategy_branch: str,
    cost_buffer: float,
) -> Candidate:
    yes = _candidate("BUY_YES", yes_ask, p_yes, cost_buffer=cost_buffer)
    no = _candidate("BUY_NO", no_ask, 1.0 - p_yes, cost_buffer=cost_buffer)
    branch = strategy_branch.strip().upper()
    if branch in {"OUTSIDER", "OUTSIDER_ONLY"}:
        return no if yes_ask >= 0.5 else yes
    if branch in {"FAVORITE", "FAVORITE_ONLY"}:
        return yes if yes_ask >= 0.5 else no
    if branch in {"COMBINED", "MIXED"}:
        return max((yes, no), key=lambda item: item.net_edge)
    raise ValueError(
        "strategy_branch must be OUTSIDER_ONLY, FAVORITE_ONLY or COMBINED"
    )


def _pnl_for_trade(
    *,
    won: bool,
    price: float,
    stake_usdc: float,
    fee_rate: float,
) -> float:
    if not won:
        return -stake_usdc
    gross_profit = stake_usdc * (1.0 - price) / price
    # Polymarket's fee is charged on the winning profit, not on the stake.
    return gross_profit - max(gross_profit, 0.0) * fee_rate


def _weighted_trade_accounting(
    *,
    won: bool,
    candidate: WeightedSideQuote,
    stake_usdc: float,
) -> tuple[float, float, float, float, float]:
    """Return budget-inclusive weighted PnL and execution accounting.

    ``stake_usdc`` is the maximum entry budget, including the modeled fee and
    slippage.  This matches the PAPER gateway's ``max_spend_usdc`` contract:
    a full-loss consumes the budget, while a win pays one USDC per share.
    """
    entry_cost_per_share = candidate.ask + candidate.cost.total_per_share
    if entry_cost_per_share <= 0.0:
        raise ValueError("weighted candidate has non-positive entry cost")
    shares = stake_usdc / entry_cost_per_share
    gross_quote = candidate.ask * shares
    fee = candidate.cost.fee_per_share * shares
    slippage = candidate.cost.slippage_per_share * shares
    spread_and_latency = (
        candidate.cost.spread_per_share + candidate.cost.latency_buffer_per_share
    ) * shares
    total_cost = gross_quote + fee + slippage + spread_and_latency
    pnl = shares - total_cost if won else -total_cost
    return (
        float(pnl),
        float(shares),
        float(gross_quote),
        float(fee),
        float(total_cost),
    )
def _oot_window_summaries(trades: list[dict[str, Any]], stake_usdc: float) -> list[dict[str, Any]]:
    """Summarize chronological OOT windows for robust experiment ranking.

    Drawdown percentages are normalized by capital deployed in the window,
    not by a single trade stake.
    """
    if not trades:
        return []
    window_count = min(3, len(trades))
    summaries: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(np.arange(len(trades)), window_count), start=1):
        if len(indices) == 0:
            continue
        pnl = np.asarray([float(trades[int(i)]["pnl"]) for i in indices], dtype=float)
        cumulative = np.cumsum(pnl)
        peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        drawdown = float(np.max(np.maximum(0.0, peak - cumulative)))
        window_invested = stake_usdc * len(indices)
        window_trades = [trades[int(i)] for i in indices]
        raw_start = window_trades[0].get("entry_time")
        raw_end = window_trades[-1].get("entry_time")
        try:
            start_timestamp = pd.Timestamp(raw_start) if raw_start is not None else None
            end_timestamp = pd.Timestamp(raw_end) if raw_end is not None else None
            if start_timestamp is not None and start_timestamp == end_timestamp:
                end_timestamp = end_timestamp + pd.Timedelta(microseconds=1)
            window_start = start_timestamp.isoformat() if start_timestamp is not None else None
            window_end = end_timestamp.isoformat() if end_timestamp is not None else None
        except (TypeError, ValueError):
            window_start = str(raw_start) if raw_start is not None else None
            window_end = str(raw_end) if raw_end is not None else None
        summaries.append({
            "window": index,
            "start": window_start,
            "end": window_end,
            "n_trades": int(len(indices)),
            "net_profit": float(pnl.sum()),
            "max_drawdown_usdc": drawdown,
            "max_drawdown_pct": drawdown / max(window_invested, 1e-9) * 100.0,
        })
    return summaries



def compute_oof_polymarket_backtest(
    frame: pd.DataFrame,
    oof_scores: Iterable[float],
    quotes: pd.DataFrame,
    *,
    strategy_branch: str = "OUTSIDER_ONLY",
    min_edge: float = DEFAULT_MIN_EDGE,
    cost_buffer: float = DEFAULT_COST_BUFFER,
    fee_rate: float | None = None,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
    outsider_max_price: float = DEFAULT_OUTSIDER_MAX_PRICE,
    stake_usdc: float = DEFAULT_STAKE_USDC,
    slippage_pct: float = 0.0,
    policy_mode: str = "LEGACY",
    p_logreg_scores: Iterable[float] | float | None = None,
    mrf_evidence_scores: Iterable[float] | float | None = None,
    market_weight: float = 0.90,
    logreg_weight: float = 0.05,
    lgbm_weight: float = 0.05,
    mrf_beta: float = 0.0,
    weighted_fee_rate: float = DEFAULT_WEIGHTED_FEE_RATE,
    weighted_fee_exponent: float = DEFAULT_WEIGHTED_FEE_EXPONENT,
    weighted_slippage_rate: float = DEFAULT_WEIGHTED_SLIPPAGE_RATE,
    execution_role: str = "TAKER",
) -> dict[str, Any]:
    """Compute comparable Polymarket PnL from strictly OOF probabilities.

    ``frame`` must contain one row per market and the canonical ``target``
    (YES=1/NO=0).  ``quotes`` contains at most one entry snapshot per market.
    Rows without an OOF probability or an entry quote are reported in the
    coverage counters and never silently treated as losses.

    ``p_logreg_scores`` is optional and, when supplied, must already be on the
    P(YES wins) axis.  Live LogReg produces P(flip), so callers should convert
    it with ``logreg_flip_to_yes_probability`` (or the equivalent canonical
    helper) before passing it here.
    """
    scores = np.asarray(list(oof_scores), dtype=float)
    if len(scores) != len(frame):
        raise ValueError("oof_scores must have the same length as frame")
    if stake_usdc <= 0:
        raise ValueError("stake_usdc must be positive")
    if not 0.0 <= slippage_pct < 1.0:
        raise ValueError("slippage_pct must be in [0, 1)")

    normalized_policy_mode = str(policy_mode or "LEGACY").strip().upper()
    weighted_mode = normalized_policy_mode in {
        "WEIGHTED",
        "WEIGHTED_SHADOW",
        "WEIGHTED_ACTIVE",
    }
    if not weighted_mode and normalized_policy_mode != "LEGACY":
        raise ValueError(
            "policy_mode must be LEGACY, WEIGHTED, WEIGHTED_SHADOW or WEIGHTED_ACTIVE"
        )
    branch = strategy_branch.strip().upper()
    if weighted_mode and branch not in {
        "OUTSIDER",
        "OUTSIDER_ONLY",
        "FAVORITE",
        "FAVORITE_ONLY",
        "COMBINED",
        "MIXED",
        "WEIGHTED",
        "WEIGHTED_ACTIVE",
    }:
        raise ValueError(
            "weighted policy backtest strategy_branch must be OUTSIDER_ONLY, "
            "FAVORITE_ONLY or COMBINED"
        )
    if weighted_mode:
        weighted_lr_scores = _normalize_score_series(
            p_logreg_scores, len(frame), "p_logreg_scores"
        )
        weighted_mrf_scores = _normalize_score_series(
            mrf_evidence_scores, len(frame), "mrf_evidence_scores"
        )
        try:
            weighted_fee_rate = float(weighted_fee_rate)
            weighted_fee_exponent = float(weighted_fee_exponent)
            weighted_slippage_rate = float(weighted_slippage_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weighted fee/slippage rates must be numeric") from exc
        if not np.isfinite(weighted_fee_rate) or weighted_fee_rate < 0.0:
            raise ValueError("weighted_fee_rate must be finite and non-negative")
        if not np.isfinite(weighted_fee_exponent) or weighted_fee_exponent < 0.0:
            raise ValueError("weighted_fee_exponent must be finite and non-negative")
        if not np.isfinite(weighted_slippage_rate) or not 0.0 <= weighted_slippage_rate < 1.0:
            raise ValueError("weighted_slippage_rate must be in [0, 1)")
        policy_config = WeightedPolicyConfig(
            market_weight=market_weight,
            logreg_weight=logreg_weight,
            lgbm_weight=lgbm_weight,
            mrf_beta=mrf_beta,
            fee_rate=weighted_fee_rate,
            fee_exponent=weighted_fee_exponent,
            slippage_rate=weighted_slippage_rate,
            execution_role=execution_role,
        )
    else:
        weighted_lr_scores = weighted_mrf_scores = None
        policy_config = None
        if fee_rate is None:
            fee_rate = DEFAULT_FEE_RATE
        try:
            fee_rate = float(fee_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("fee_rate must be numeric") from exc
        if not np.isfinite(fee_rate) or fee_rate < 0.0:
            raise ValueError("fee_rate must be finite and non-negative")
    accounting_model = (
        "POLYMARKET_PRICE_DEPENDENT_BUDGET"
        if weighted_mode
        else "LEGACY_WIN_PROFIT"
    )
    effective_fee_rate = (
        float(weighted_fee_rate) if weighted_mode else float(fee_rate)
    )

    base = frame.reset_index(drop=True).copy()
    base["_oof_index"] = np.arange(len(base), dtype=int)
    base["_p_yes"] = scores
    base["market_id"] = base["market_id"].astype(str)
    quote_frame = quotes.copy() if quotes is not None else pd.DataFrame()
    if quote_frame.empty:
        merged = base.iloc[0:0].copy()
    else:
        quote_frame["market_id"] = quote_frame["market_id"].astype(str)
        merged = base.merge(
            quote_frame.drop_duplicates("market_id", keep="first"),
            on="market_id",
            how="inner",
            suffixes=("", "_quote"),
        )

    trades: list[dict[str, Any]] = []
    eligible_rows = 0
    coverage_reasons = Counter()
    coverage_reasons["missing_oof"] = int(np.isnan(scores).sum())
    coverage_reasons["missing_quote"] = max(int(len(base) - len(merged)), 0)
    for _, row in merged.iterrows():
        source_index = int(row["_oof_index"])
        raw_p_yes = row.get("_p_yes")
        p_yes = _as_float(raw_p_yes)
        if p_yes is None:
            if raw_p_yes is not None and not pd.isna(raw_p_yes):
                coverage_reasons["invalid_oof"] += 1
            continue
        if not 0.0 <= p_yes <= 1.0:
            coverage_reasons["invalid_oof"] += 1
            continue
        yes_ask, no_ask = _quote_prices(row)
        if yes_ask is None or no_ask is None:
            coverage_reasons["invalid_quote"] += 1
            continue
        weighted_candidate: WeightedSideQuote | None = None
        if weighted_mode:
            p_market_yes = _market_mid_price(row, yes_ask, no_ask)
            p_logreg_yes = (
                _as_float(weighted_lr_scores[source_index])
                if weighted_lr_scores is not None
                else None
            )
            mrf_evidence = (
                _as_float(weighted_mrf_scores[source_index])
                if weighted_mrf_scores is not None
                else None
            )
            if weighted_lr_scores is not None and p_logreg_yes is None:
                coverage_reasons["invalid_logreg"] += 1
            if weighted_mrf_scores is not None and mrf_evidence is None:
                coverage_reasons["invalid_mrf_evidence"] += 1
            selection = select_weighted_side(
                p_market_yes=p_market_yes,
                p_logreg_yes=p_logreg_yes,
                p_lgbm_yes=p_yes,
                yes_ask=yes_ask,
                no_ask=no_ask,
                config=policy_config,
                mrf_evidence=mrf_evidence,
                min_net_ev=0.0,
                fee_source="BACKTEST_CONFIG",
                spread=_as_float(row.get("spread"), 0.0) or 0.0,
            )
            # The combined policy normally takes the best side.  Branch
            # reports still need to answer the old diagnostic question
            # "what if we traded only favorites/outsiders?", so constrain the
            # same cost-aware quotes before applying the branch price cap.
            if branch in {"OUTSIDER", "OUTSIDER_ONLY"}:
                weighted_candidate = (
                    selection.no_quote if yes_ask >= 0.5 else selection.yes_quote
                )
            elif branch in {"FAVORITE", "FAVORITE_ONLY"}:
                weighted_candidate = (
                    selection.yes_quote if yes_ask >= 0.5 else selection.no_quote
                )
            else:
                weighted_candidate = selection.selected
            if weighted_candidate is None:
                coverage_reasons["no_positive_weighted_ev"] += 1
                continue
            if weighted_candidate.net_ev_per_share < 0.0:
                coverage_reasons["no_positive_weighted_ev"] += 1
                continue
            candidate = Candidate(
                weighted_candidate.side,
                weighted_candidate.ask,
                weighted_candidate.p_win,
                weighted_candidate.gross_ev_per_share,
                weighted_candidate.net_ev_per_share,
            )
            price_cap = (
                outsider_max_price
                if branch in {"OUTSIDER", "OUTSIDER_ONLY"}
                else max_price
            )
        else:
            candidate = _select_candidate(
                p_yes,
                yes_ask,
                no_ask,
                strategy_branch=strategy_branch,
                cost_buffer=cost_buffer,
            )
            price_cap = outsider_max_price if branch in {"OUTSIDER", "OUTSIDER_ONLY"} else max_price
        if not min_price <= candidate.price <= min(max_price, price_cap):
            coverage_reasons["price_out_of_bounds"] += 1
            continue
        eligible_rows += 1
        if candidate.net_edge < min_edge:
            coverage_reasons["insufficient_edge"] += 1
            continue

        executed_price = (
            candidate.price
            if weighted_mode
            else min(candidate.price * (1.0 + slippage_pct), 0.99)
        )
        outcome = str(
            row.get("final_outcome") or ("YES" if row.get("target") == 1 else "NO")
        ).strip().upper()
        won = (candidate.side == "BUY_YES" and outcome == "YES") or (
            candidate.side == "BUY_NO" and outcome == "NO"
        )
        trade_pnl = None
        trade_shares = None
        gross_quote_usdc = None
        fee_usdc = None
        total_cost_usdc = None
        if weighted_mode:
            assert weighted_candidate is not None
            (
                trade_pnl,
                trade_shares,
                gross_quote_usdc,
                fee_usdc,
                total_cost_usdc,
            ) = _weighted_trade_accounting(
                won=won,
                candidate=weighted_candidate,
                stake_usdc=stake_usdc,
            )
        else:
            trade_pnl = _pnl_for_trade(
                won=won,
                price=executed_price,
                stake_usdc=stake_usdc,
                fee_rate=fee_rate,
            )
        trades.append(
            {
                "market_id": str(row["market_id"]),
                "asset": str(row.get("asset") or ""),
                "side": candidate.side,
                "price": float(executed_price),
                "p_win": candidate.p_win,
                "gross_edge": candidate.gross_edge,
                "net_edge": candidate.net_edge,
                "outcome": outcome,
                "won": bool(won),
                "pnl": trade_pnl,
                "shares": trade_shares,
                "gross_quote_usdc": gross_quote_usdc,
                "fee_usdc": fee_usdc,
                "total_cost_usdc": total_cost_usdc,
                "policy_mode": normalized_policy_mode,
                "entry_time": row.get("recorded_at") or row.get("market_start"),
                "time_left_min": _as_float(row.get("time_left_min")),
                "price_bucket": _price_bucket(executed_price),
                "phase": _phase_bucket(
                    _as_float(row.get("time_left_min")), row.get("vol_regime")
                ),
            }
        )

    trades.sort(key=lambda item: str(item.get("entry_time") or ""))
    if not trades:
        return {
            "strategy_branch": strategy_branch.upper(),
            "policy_mode": normalized_policy_mode,
            "accounting_model": accounting_model,
            "fee_model": (
                "POLYMARKET_PRICE_DEPENDENT"
                if weighted_mode
                else "LEGACY_WIN_PROFIT"
            ),
            "fee_rate": effective_fee_rate,
            "fee_exponent": (float(weighted_fee_exponent) if weighted_mode else 1.0),
            "execution_role": (
                str(execution_role or "TAKER").strip().upper()
                if weighted_mode
                else None
            ),
            "n_markets": int(len(base)),
            "n_quotes": int(len(merged)),
            "n_oof": int(np.isfinite(scores).sum()),
            "n_eligible": int(eligible_rows),
            "n_trades": 0,
            "win_rate": 0.0,
            "total_invested": 0.0,
            "stake_usdc": float(stake_usdc),
            "net_profit": 0.0,
            "roi_pct": 0.0,
            "avg_edge": 0.0,
            "avg_net_edge": 0.0,
            "avg_entry_price": None,
            "max_drawdown_usdc": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": None,
            "profit_factor": 0.0,
            "coverage_pct": round(len(merged) / len(base) * 100, 2) if len(base) else 0.0,
            "coverage_reasons": dict(coverage_reasons),
            "slices": [],
            "equity_curve": [],
            "trades": [],
            "oot_windows": [],
        }

    pnl = np.asarray([float(item["pnl"]) for item in trades], dtype=float)
    invested = stake_usdc * len(trades)
    cumulative = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdowns = np.maximum(0.0, running_peak - cumulative)
    max_dd_usdc = float(drawdowns.max())
    # A drawdown percentage is relative to capital deployed by this backtest,
    # not to the stake of a single trade.
    max_dd_pct = float(max_dd_usdc / max(invested, 1e-9) * 100.0)
    std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = float(np.mean(pnl) / std * np.sqrt(len(pnl))) if std > 0 else None
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))

    equity_curve = [
        {
            "trade_index": i,
            "market_id": item["market_id"],
            "side": item["side"],
            "pnl": round(float(cumulative[i]), 8),
            "trade_pnl": round(float(item["pnl"]), 8),
            "price": round(float(item["price"]), 6),
            "net_edge": round(float(item["net_edge"]), 6),
            "outcome": "WIN" if item["won"] else "LOSS",
            "entry_time": item["entry_time"].isoformat()
            if hasattr(item["entry_time"], "isoformat")
            else str(item["entry_time"]),
        }
        for i, item in enumerate(trades)
    ]

    slices: list[dict[str, Any]] = []
    for dimension, key in (("DIRECTION", "side"), ("PRICE", "price_bucket"), ("PHASE", "phase")):
        for bucket, items in sorted(
            ((value, [item for item in trades if item[key] == value]) for value in {item[key] for item in trades}),
            key=lambda pair: str(pair[0]),
        ):
            bucket_pnl = sum(float(item["pnl"]) for item in items)
            slices.append(
                {
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "trades": len(items),
                    "net_pnl": round(bucket_pnl, 8),
                    "roi_pct": round(bucket_pnl / (stake_usdc * len(items)) * 100.0, 4),
                    "win_rate_pct": round(sum(item["won"] for item in items) / len(items) * 100.0, 4),
                    "avg_entry_price": round(float(np.mean([item["price"] for item in items])), 6),
                    "avg_edge": round(float(np.mean([item["gross_edge"] for item in items])), 6),
                }
            )

    return {
        "strategy_branch": strategy_branch.upper(),
        "policy_mode": normalized_policy_mode,
        "accounting_model": accounting_model,
        "fee_model": (
            "POLYMARKET_PRICE_DEPENDENT"
            if weighted_mode
            else "LEGACY_WIN_PROFIT"
        ),
        "fee_rate": effective_fee_rate,
        "fee_exponent": (float(weighted_fee_exponent) if weighted_mode else 1.0),
        "execution_role": (
            str(execution_role or "TAKER").strip().upper()
            if weighted_mode
            else None
        ),
        "n_markets": int(len(base)),
        "n_quotes": int(len(merged)),
        "n_oof": int(np.isfinite(scores).sum()),
        "n_eligible": int(eligible_rows),
        "n_trades": len(trades),
        "win_rate": float(np.mean([item["won"] for item in trades])),
        "total_invested": float(invested),
        "stake_usdc": float(stake_usdc),
        "net_profit": float(pnl.sum()),
        "roi_pct": float(pnl.sum() / invested * 100.0),
        "avg_edge": float(np.mean([item["gross_edge"] for item in trades])),
        "avg_net_edge": float(np.mean([item["net_edge"] for item in trades])),
        "avg_entry_price": float(np.mean([item["price"] for item in trades])),
        "max_drawdown_usdc": max_dd_usdc,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "coverage_pct": round(len(merged) / len(base) * 100, 2) if len(base) else 0.0,
        "coverage_reasons": dict(coverage_reasons),
        "slices": slices,
        "equity_curve": equity_curve,
        "trades": trades,
        "oot_windows": _oot_window_summaries(trades, stake_usdc),
    }

def aggregate_stored_polymarket_backtests(
    regime_results: Iterable[dict[str, Any]],
    *,
    strategy_branch: str,
) -> dict[str, Any]:
    """Combine persisted per-volatility-regime OOF summaries for the UI.

    Training stores each regime independently so its PnL can be audited.  The
    dashboard needs one portfolio number, therefore this function replays the
    persisted ``trade_pnl`` values in entry-time order instead of adding
    already-cumulative equity values.
    """
    results = [r for r in regime_results if isinstance(r, dict)]
    branch = strategy_branch.strip().upper()
    curve_items: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    n_markets = n_quotes = n_oof = n_eligible = n_trades = 0
    net_profit = total_invested = edge_sum = net_edge_sum = price_sum = 0.0
    wins = 0.0
    weighted_stake = 0.0
    weighted_stake_count = 0
    coverage_reasons = Counter()
    policy_modes: set[str] = set()
    accounting_models: set[str] = set()
    fee_models: set[str] = set()
    execution_roles: set[str] = set()
    fee_rates: list[float] = []
    fee_exponents: list[float] = []
    for result in results:
        for value, bucket in (
            (result.get("policy_mode"), policy_modes),
            (result.get("accounting_model"), accounting_models),
            (result.get("fee_model"), fee_models),
            (result.get("execution_role"), execution_roles),
        ):
            if value is not None and str(value).strip():
                bucket.add(str(value).strip().upper())
        try:
            result_fee_rate = float(result.get("fee_rate"))
            if np.isfinite(result_fee_rate):
                fee_rates.append(result_fee_rate)
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            result_fee_exponent = float(result.get("fee_exponent"))
            if np.isfinite(result_fee_exponent):
                fee_exponents.append(result_fee_exponent)
        except (TypeError, ValueError, OverflowError):
            pass
        n_markets += int(result.get("n_markets") or 0)
        n_quotes += int(result.get("n_quotes") or 0)
        n_oof += int(result.get("n_oof") or 0)
        n_eligible += int(result.get("n_eligible") or 0)
        count = int(result.get("n_trades") or 0)
        n_trades += count
        net_profit += float(result.get("net_profit") or 0.0)
        invested_val = float(result.get("total_invested") or 0.0)
        persisted_stake = result.get("stake_usdc")
        try:
            parsed_stake = float(persisted_stake) if persisted_stake is not None else None
        except (TypeError, ValueError):
            parsed_stake = None
        if parsed_stake is not None and (
            not np.isfinite(parsed_stake) or parsed_stake <= 0.0
        ):
            parsed_stake = None
        if parsed_stake is not None and count > 0:
            weighted_stake += parsed_stake * count
            weighted_stake_count += count
        # Older persisted summaries may omit stake_usdc and/or have a zero
        # total_invested even though trades were recorded.  Their historical
        # contract used the one-USDC default stake, so keep ROI finite instead
        # of silently reporting zero invested capital.
        if count > 0 and invested_val <= 0.0:
            invested_val = (parsed_stake or DEFAULT_STAKE_USDC) * count
        total_invested += invested_val
        edge_sum += float(result.get("avg_edge") or 0.0) * count
        net_edge_sum += float(result.get("avg_net_edge") or 0.0) * count
        price_sum += float(result.get("avg_entry_price") or 0.0) * count
        wins += float(result.get("win_rate") or 0.0) * count
        for reason, reason_count in (result.get("coverage_reasons") or {}).items():
            try:
                coverage_reasons[str(reason)] += int(reason_count)
            except (TypeError, ValueError):
                continue
        slices.extend(result.get("slices") or [])
        curve_items.extend(result.get("equity_curve") or [])

    curve_items.sort(key=lambda item: str(item.get("entry_time") or ""))
    cumulative = 0.0
    equity_curve: list[dict[str, Any]] = []
    for index, item in enumerate(curve_items):
        trade_pnl = float(item.get("trade_pnl") or 0.0)
        cumulative += trade_pnl
        equity_curve.append({
            **item,
            "trade_index": index,
            "pnl": round(cumulative, 8),
            "trade_pnl": round(trade_pnl, 8),
        })
    max_dd = 0.0
    if equity_curve:
        equity = np.asarray([float(item["pnl"]) for item in equity_curve])
        peak = np.maximum.accumulate(np.maximum(equity, 0.0))
        max_dd = float(np.max(np.maximum(0.0, peak - equity)))
    pnl_values = np.asarray([float(item.get("trade_pnl") or 0.0) for item in equity_curve])
    std = float(np.std(pnl_values, ddof=1)) if len(pnl_values) > 1 else 0.0
    sharpe = float(np.mean(pnl_values) / std * np.sqrt(len(pnl_values))) if std > 0 else None
    gross_profit = float(pnl_values[pnl_values > 0].sum()) if len(pnl_values) else 0.0
    gross_loss = float(abs(pnl_values[pnl_values < 0].sum())) if len(pnl_values) else 0.0
    coverage = round(n_quotes / n_markets * 100.0, 2) if n_markets else 0.0
    stake_per_trade = (
        weighted_stake / weighted_stake_count
        if weighted_stake_count
        else (total_invested / n_trades if n_trades else 1.0)
    )

    def _single_or_mixed(values: set[str]) -> str | None:
        if not values:
            return None
        return next(iter(values)) if len(values) == 1 else "MIXED"

    return {
        "strategy_branch": branch,
        "policy_mode": _single_or_mixed(policy_modes),
        "accounting_model": _single_or_mixed(accounting_models),
        "fee_model": _single_or_mixed(fee_models),
        "fee_rate": float(np.mean(fee_rates)) if fee_rates else None,
        "fee_exponent": float(np.mean(fee_exponents)) if fee_exponents else None,
        "execution_role": _single_or_mixed(execution_roles),
        "n_markets": n_markets,
        "n_quotes": n_quotes,
        "n_oof": n_oof,
        "n_eligible": n_eligible,
        "n_trades": n_trades,
        "win_rate": wins / n_trades if n_trades else 0.0,
        "total_invested": total_invested,
        "stake_usdc": stake_per_trade,
        "net_profit": net_profit,
        "roi_pct": net_profit / total_invested * 100.0 if total_invested else 0.0,
        "avg_edge": edge_sum / n_trades if n_trades else 0.0,
        "avg_net_edge": net_edge_sum / n_trades if n_trades else 0.0,
        "avg_entry_price": price_sum / n_trades if n_trades else None,
        "max_drawdown_usdc": max_dd,
        # Normalize by the total capital deployed across all persisted trades,
        # not by one representative trade stake.
        "max_drawdown_pct": max_dd / max(total_invested, 1e-9) * 100.0 if n_trades else 0.0,
        "sharpe_ratio": sharpe,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "coverage_pct": coverage,
        "coverage_reasons": dict(coverage_reasons),
        "slices": slices,
        "equity_curve": equity_curve,
        "trades": [],
    }

async def load_market_entry_quotes(
    db: AsyncSession,
    market_starts: pd.DataFrame,
    *,
    chunk_size: int = 4000,
) -> pd.DataFrame:
    """Load the first executable snapshot at/after each market start.

    The query is chunked to avoid PostgreSQL's bind-parameter limit when the
    training dataset contains tens of thousands of markets.
    """
    if market_starts is None or market_starts.empty:
        return pd.DataFrame()
    starts = market_starts[["market_id", "market_start"]].copy()
    starts["market_id"] = starts["market_id"].astype(str)
    starts["market_start"] = pd.to_datetime(starts["market_start"], utc=True)
    ids = starts["market_id"].drop_duplicates().tolist()
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(ids), chunk_size):
        chunk = ids[offset : offset + chunk_size]
        stmt = (
            select(
                MarketSnapshot.market_id,
                MarketSnapshot.asset,
                MarketSnapshot.recorded_at,
                MarketSnapshot.time_left_min,
                MarketSnapshot.mid_price,
                MarketSnapshot.spread,
                MarketSnapshot.best_bid,
                MarketSnapshot.best_ask,
                MarketSnapshot.final_outcome,
            )
            .where(
                MarketSnapshot.market_id.in_(chunk),
            )
            .order_by(MarketSnapshot.market_id, MarketSnapshot.recorded_at)
        )
        result = await db.execute(stmt)
        rows.extend(dict(row._mapping) for row in result.fetchall())
    if not rows:
        return pd.DataFrame()
    quotes = pd.DataFrame(rows)
    quotes["market_id"] = quotes["market_id"].astype(str)
    quotes["recorded_at"] = pd.to_datetime(quotes["recorded_at"], utc=True)
    quotes = quotes.merge(starts, on="market_id", how="inner")
    quotes = quotes[quotes["recorded_at"] >= quotes["market_start"]]
    return (
        quotes.sort_values(["market_id", "recorded_at"])
        .drop_duplicates("market_id", keep="first")
        .drop(columns=["market_start"])
        .reset_index(drop=True)
    )
