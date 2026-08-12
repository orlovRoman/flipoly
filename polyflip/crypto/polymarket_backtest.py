"""Polymarket OOF backtesting helpers for canonical LightGBM models.

The LightGBM candle backtester historically measured the next Binance return.
That is useful as a diagnostic, but it is not the trading PnL of a binary
Polymarket token.  This module keeps the accounting separate and evaluates
OOF probabilities against the actual YES/NO quote and the canonical market
resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import MarketSnapshot


DEFAULT_MIN_EDGE = 0.04
DEFAULT_COST_BUFFER = 0.02
DEFAULT_FEE_RATE = 0.002
DEFAULT_MIN_PRICE = 0.05
DEFAULT_MAX_PRICE = 0.95
DEFAULT_OUTSIDER_MAX_PRICE = 0.45
DEFAULT_STAKE_USDC = 1.0


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


def compute_oof_polymarket_backtest(
    frame: pd.DataFrame,
    oof_scores: Iterable[float],
    quotes: pd.DataFrame,
    *,
    strategy_branch: str = "OUTSIDER_ONLY",
    min_edge: float = DEFAULT_MIN_EDGE,
    cost_buffer: float = DEFAULT_COST_BUFFER,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
    outsider_max_price: float = DEFAULT_OUTSIDER_MAX_PRICE,
    stake_usdc: float = DEFAULT_STAKE_USDC,
    slippage_pct: float = 0.0,
) -> dict[str, Any]:
    """Compute comparable Polymarket PnL from strictly OOF probabilities.

    ``frame`` must contain one row per market and the canonical ``target``
    (YES=1/NO=0).  ``quotes`` contains at most one entry snapshot per market.
    Rows without an OOF probability or an entry quote are reported in the
    coverage counters and never silently treated as losses.
    """
    scores = np.asarray(list(oof_scores), dtype=float)
    if len(scores) != len(frame):
        raise ValueError("oof_scores must have the same length as frame")
    if stake_usdc <= 0:
        raise ValueError("stake_usdc must be positive")
    if not 0.0 <= slippage_pct < 1.0:
        raise ValueError("slippage_pct must be in [0, 1)")

    base = frame.reset_index(drop=True).copy()
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
    for _, row in merged.iterrows():
        p_yes = _as_float(row.get("_p_yes"))
        if p_yes is None or not 0.0 <= p_yes <= 1.0:
            continue
        yes_ask, no_ask = _quote_prices(row)
        if yes_ask is None or no_ask is None:
            continue
        candidate = _select_candidate(
            p_yes,
            yes_ask,
            no_ask,
            strategy_branch=strategy_branch,
            cost_buffer=cost_buffer,
        )
        branch = strategy_branch.strip().upper()
        price_cap = outsider_max_price if branch in {"OUTSIDER", "OUTSIDER_ONLY"} else max_price
        if not min_price <= candidate.price <= min(max_price, price_cap):
            continue
        eligible_rows += 1
        if candidate.net_edge < min_edge:
            continue

        executed_price = min(candidate.price * (1.0 + slippage_pct), 0.99)
        outcome = str(row.get("final_outcome") or ("YES" if row.get("target") == 1 else "NO"))
        won = (candidate.side == "BUY_YES" and outcome == "YES") or (
            candidate.side == "BUY_NO" and outcome == "NO"
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
                "pnl": _pnl_for_trade(
                    won=won,
                    price=executed_price,
                    stake_usdc=stake_usdc,
                    fee_rate=fee_rate,
                ),
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
            "n_markets": int(len(base)),
            "n_quotes": int(len(merged)),
            "n_oof": int(np.isfinite(scores).sum()),
            "n_eligible": int(eligible_rows),
            "n_trades": 0,
            "win_rate": 0.0,
            "total_invested": 0.0,
            "net_profit": 0.0,
            "roi_pct": 0.0,
            "avg_edge": 0.0,
            "avg_net_edge": 0.0,
            "avg_entry_price": None,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": None,
            "profit_factor": 0.0,
            "coverage_pct": round(len(merged) / len(base) * 100, 2) if len(base) else 0.0,
            "slices": [],
            "equity_curve": [],
            "trades": [],
        }

    pnl = np.asarray([float(item["pnl"]) for item in trades], dtype=float)
    invested = stake_usdc * len(trades)
    cumulative = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdowns = np.maximum(0.0, running_peak - cumulative)
    max_dd_pct = float(drawdowns.max() / max(stake_usdc, 1e-9) * 100.0)
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
        "n_markets": int(len(base)),
        "n_quotes": int(len(merged)),
        "n_oof": int(np.isfinite(scores).sum()),
        "n_eligible": int(eligible_rows),
        "n_trades": len(trades),
        "win_rate": float(np.mean([item["won"] for item in trades])),
        "total_invested": float(invested),
        "net_profit": float(pnl.sum()),
        "roi_pct": float(pnl.sum() / invested * 100.0),
        "avg_edge": float(np.mean([item["gross_edge"] for item in trades])),
        "avg_net_edge": float(np.mean([item["net_edge"] for item in trades])),
        "avg_entry_price": float(np.mean([item["price"] for item in trades])),
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "coverage_pct": round(len(merged) / len(base) * 100, 2) if len(base) else 0.0,
        "slices": slices,
        "equity_curve": equity_curve,
        "trades": trades,
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
    for result in results:
        n_markets += int(result.get("n_markets") or 0)
        n_quotes += int(result.get("n_quotes") or 0)
        n_oof += int(result.get("n_oof") or 0)
        n_eligible += int(result.get("n_eligible") or 0)
        count = int(result.get("n_trades") or 0)
        n_trades += count
        net_profit += float(result.get("net_profit") or 0.0)
        total_invested += float(result.get("total_invested") or 0.0)
        edge_sum += float(result.get("avg_edge") or 0.0) * count
        net_edge_sum += float(result.get("avg_net_edge") or 0.0) * count
        price_sum += float(result.get("avg_entry_price") or 0.0) * count
        wins += float(result.get("win_rate") or 0.0) * count
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
    return {
        "strategy_branch": branch,
        "n_markets": n_markets,
        "n_quotes": n_quotes,
        "n_oof": n_oof,
        "n_eligible": n_eligible,
        "n_trades": n_trades,
        "win_rate": wins / n_trades if n_trades else 0.0,
        "total_invested": total_invested,
        "net_profit": net_profit,
        "roi_pct": net_profit / total_invested * 100.0 if total_invested else 0.0,
        "avg_edge": edge_sum / n_trades if n_trades else 0.0,
        "avg_net_edge": net_edge_sum / n_trades if n_trades else 0.0,
        "avg_entry_price": price_sum / n_trades if n_trades else None,
        "max_drawdown_pct": max_dd / max(total_invested / max(n_trades, 1), 1e-9) * 100.0 if n_trades else 0.0,
        "sharpe_ratio": sharpe,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "coverage_pct": coverage,
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
                MarketSnapshot.final_outcome.in_(("YES", "NO")),
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
