"""Joint LightGBM threshold search on saved OOF predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from polyflip.crypto.polymarket_backtest import (
    DEFAULT_COST_BUFFER,
    DEFAULT_FEE_RATE,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_EDGE,
    DEFAULT_MIN_PRICE,
    DEFAULT_OUTSIDER_MAX_PRICE,
    DEFAULT_STAKE_USDC,
    _as_float,
    _oot_window_summaries,
    _phase_bucket,
    _pnl_for_trade,
    _price_bucket,
    _quote_prices,
)

TARGET_COVERAGES: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)


@dataclass(frozen=True)
class ThresholdPair:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("thresholds must be finite")
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= lower < upper <= 1")


def raw_opinion_for_score(score: float) -> str:
    return "UP" if float(score) >= 0.5 else "DOWN"


def classify_scores(scores: Iterable[float], *, lower: float, upper: float) -> np.ndarray:
    pair = ThresholdPair(float(lower), float(upper))
    values = np.asarray(list(scores), dtype=float)
    result = np.full(values.shape, "NONE", dtype=object)
    finite = np.isfinite(values)
    result[finite & (values <= pair.lower)] = "DOWN"
    result[finite & (values >= pair.upper)] = "UP"
    return result


def _quantile_pair(scores: np.ndarray, target_coverage: float) -> ThresholdPair:
    target = float(target_coverage)
    if not 0.0 < target < 1.0:
        raise ValueError("target coverage must be between 0 and 1")
    valid = scores[np.isfinite(scores)]
    if len(valid) < 2:
        return ThresholdPair(0.49, 0.51)
    tail = target / 2.0
    lower = float(np.quantile(valid, tail, method="linear"))
    upper = float(np.quantile(valid, 1.0 - tail, method="linear"))
    if lower >= upper:
        ordered = np.sort(np.unique(valid))
        if len(ordered) < 2:
            return ThresholdPair(0.49, 0.51)
        lower_i = max(0, min(len(ordered) - 2, int(len(ordered) * tail)))
        upper_i = min(len(ordered) - 1, max(lower_i + 1, int(len(ordered) * (1.0 - tail))))
        lower, upper = float(ordered[lower_i]), float(ordered[upper_i])
    return ThresholdPair(max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper)))


def threshold_grid(
    scores: Iterable[float],
    *,
    target_coverage: float,
    quantile_steps: int = 5,
    tolerance: float = 0.10,
) -> list[ThresholdPair]:
    """Create a bounded asymmetric grid around one target coverage."""
    values = np.asarray(list(scores), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [ThresholdPair(0.49, 0.51)]
    pairs: dict[tuple[float, float], ThresholdPair] = {}
    canonical = _quantile_pair(values, target_coverage)
    pairs[(round(canonical.lower, 8), round(canonical.upper, 8))] = canonical
    target = float(target_coverage)
    half = target / 2.0
    low = np.linspace(max(0.0, half - tolerance), min(0.49, half + tolerance), max(4, int(quantile_steps)))
    high = np.linspace(max(0.51, 1.0 - half - tolerance), min(1.0, 1.0 - half + tolerance), max(4, int(quantile_steps)))
    for lower_q in low:
        for upper_q in high:
            if lower_q >= upper_q:
                continue
            try:
                pair = ThresholdPair(
                    float(np.quantile(values, lower_q, method="linear")),
                    float(np.quantile(values, upper_q, method="linear")),
                )
            except ValueError:
                continue
            pairs[(round(pair.lower, 8), round(pair.upper, 8))] = pair
    return list(pairs.values())


def _max_drawdown(pnls: list[float]) -> tuple[float, float]:
    if not pnls:
        return 0.0, 0.0
    values = np.asarray(pnls, dtype=float)
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdown = float(np.max(np.maximum(0.0, peak - cumulative)))
    return drawdown, drawdown / max(float(len(values)), 1.0) * 100.0


def _slice_summary(trades: list[dict[str, Any]], key: str, stake_usdc: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for bucket in sorted({str(item.get(key)) for item in trades}):
        items = [item for item in trades if str(item.get(key)) == bucket]
        pnls = [float(item["pnl"]) for item in items]
        result.append({
            "dimension": "ROLE" if key == "market_role" else "DIRECTION",
            "bucket": bucket,
            "trades": len(items),
            "net_pnl": round(sum(pnls), 8),
            "roi_pct": round(sum(pnls) / max(stake_usdc * len(items), 1e-9) * 100.0, 4),
            "win_rate_pct": round(sum(bool(item["won"]) for item in items) / len(items) * 100.0, 4),
        })
    return result


def evaluate_thresholded_polymarket(
    frame: pd.DataFrame,
    raw_scores: Iterable[float],
    calibrated_scores: Iterable[float],
    quotes: pd.DataFrame,
    *,
    lower: float,
    upper: float,
    strategy_branch: str = "COMBINED",
    summary_only: bool = False,
    min_edge: float = DEFAULT_MIN_EDGE,
    cost_buffer: float = DEFAULT_COST_BUFFER,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
    outsider_max_price: float = DEFAULT_OUTSIDER_MAX_PRICE,
    stake_usdc: float = DEFAULT_STAKE_USDC,
    slippage_pct: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one joint threshold pair against executable Polymarket quotes."""
    pair = ThresholdPair(float(lower), float(upper))
    raw = np.asarray(list(raw_scores), dtype=float)
    calibrated = np.asarray(list(calibrated_scores), dtype=float)
    if len(raw) != len(frame) or len(calibrated) != len(frame):
        raise ValueError("score arrays must align with frame")
    if stake_usdc <= 0:
        raise ValueError("stake_usdc must be positive")
    actions = classify_scores(raw, lower=pair.lower, upper=pair.upper)
    valid_oof = np.isfinite(raw) & np.isfinite(calibrated)
    base = frame.reset_index(drop=True).copy()
    base["_orig_idx"] = np.arange(len(base))
    base["market_id"] = base["market_id"].astype(str)
    base["_p_up"] = calibrated
    quote_frame = quotes.copy() if quotes is not None else pd.DataFrame()
    if quote_frame.empty:
        merged = base.iloc[0:0].copy()
    else:
        quote_frame["market_id"] = quote_frame["market_id"].astype(str)
        merged = base.merge(quote_frame.drop_duplicates("market_id", keep="first"), on="market_id", how="inner", suffixes=("", "_quote"))

    trades: list[dict[str, Any]] = []
    coverage_reasons: dict[str, int] = {"missing_oof": int((~valid_oof).sum()), "missing_quote": max(int(len(base) - len(merged)), 0)}
    signals = up_signals = down_signals = eligible_rows = 0
    branch = strategy_branch.strip().upper()
    for _, row in merged.iterrows():
        index = int(row["_orig_idx"])
        action = str(actions[index])
        if action == "NONE" or not valid_oof[index]:
            continue
        signals += 1
        up_signals += action == "UP"
        down_signals += action == "DOWN"
        p_up = _as_float(row.get("_p_up"))
        yes_ask, no_ask = _quote_prices(row)
        if p_up is None or not 0.0 <= p_up <= 1.0:
            coverage_reasons["invalid_calibrated_oof"] = coverage_reasons.get("invalid_calibrated_oof", 0) + 1
            continue
        if yes_ask is None or no_ask is None:
            coverage_reasons["invalid_quote"] = coverage_reasons.get("invalid_quote", 0) + 1
            continue
        side = "BUY_YES" if action == "UP" else "BUY_NO"
        price = yes_ask if side == "BUY_YES" else no_ask
        outsider = (side == "BUY_YES" and yes_ask < 0.5) or (side == "BUY_NO" and yes_ask >= 0.5)
        if branch in {"OUTSIDER", "OUTSIDER_ONLY"} and not outsider:
            coverage_reasons["branch_filtered"] = coverage_reasons.get("branch_filtered", 0) + 1
            continue
        if branch in {"FAVORITE", "FAVORITE_ONLY"} and outsider:
            coverage_reasons["branch_filtered"] = coverage_reasons.get("branch_filtered", 0) + 1
            continue
        price_cap = outsider_max_price if branch in {"OUTSIDER", "OUTSIDER_ONLY"} else max_price
        if not min_price <= price <= min(max_price, price_cap):
            coverage_reasons["price_out_of_bounds"] = coverage_reasons.get("price_out_of_bounds", 0) + 1
            continue
        p_win = p_up if side == "BUY_YES" else 1.0 - p_up
        gross_edge = float(p_win - price)
        net_edge = gross_edge - cost_buffer
        eligible_rows += 1
        if net_edge < min_edge:
            coverage_reasons["insufficient_edge"] = coverage_reasons.get("insufficient_edge", 0) + 1
            continue
        executed_price = min(price * (1.0 + slippage_pct), 0.99)
        outcome = str(row.get("final_outcome") or ("YES" if row.get("target") == 1 else "NO"))
        won = (side == "BUY_YES" and outcome == "YES") or (side == "BUY_NO" and outcome == "NO")
        trades.append({
            "market_id": str(row["market_id"]),
            "asset": str(row.get("asset") or ""),
            "side": side,
            "action": action,
            "market_role": "OUTSIDER" if outsider else "FAVORITE",
            "price": float(executed_price),
            "p_win": float(p_win),
            "gross_edge": gross_edge,
            "net_edge": net_edge,
            "outcome": outcome,
            "won": bool(won),
            "pnl": _pnl_for_trade(won=won, price=executed_price, stake_usdc=stake_usdc, fee_rate=fee_rate),
            "entry_time": row.get("recorded_at") or row.get("market_start"),
            "time_left_min": _as_float(row.get("time_left_min")),
            "price_bucket": _price_bucket(executed_price),
            "phase": _phase_bucket(_as_float(row.get("time_left_min")), row.get("vol_regime")),
        })
    trades.sort(key=lambda item: str(item.get("entry_time") or ""))
    pnls = [float(item["pnl"]) for item in trades]
    max_dd, max_dd_pct = _max_drawdown(pnls)
    invested = stake_usdc * len(trades)
    n_valid = int(valid_oof.sum())
    n_signals = int((actions != "NONE")[valid_oof].sum())
    res = {
        "lower_threshold": pair.lower,
        "upper_threshold": pair.upper,
        "strategy_branch": branch,
        "n_markets": int(len(base)),
        "n_quotes": int(len(merged)),
        "n_oof": n_valid,
        "n_signals": n_signals,
        "up_signals": int(up_signals),
        "down_signals": int(down_signals),
        "none_signals": max(n_valid - n_signals, 0),
        "coverage_pct": round(n_signals / n_valid * 100.0, 2) if n_valid else 0.0,
        "none_pct": round(max(n_valid - n_signals, 0) / n_valid * 100.0, 2) if n_valid else 0.0,
        "n_eligible": int(eligible_rows),
        "n_trades": len(trades),
        "win_rate": float(np.mean([item["won"] for item in trades])) if trades else 0.0,
        "total_invested": float(invested),
        "stake_usdc": float(stake_usdc),
        "net_profit": float(sum(pnls)),
        "roi_pct": float(sum(pnls) / invested * 100.0) if invested else 0.0,
        "max_drawdown_usdc": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "coverage_reasons": coverage_reasons,
    }
    if not summary_only:
        res["slices"] = _slice_summary(trades, "action", stake_usdc) + _slice_summary(trades, "market_role", stake_usdc)
        res["oot_windows"] = _oot_window_summaries(trades, stake_usdc)
        res["trades"] = trades
    return res


def _economic_score(result: dict[str, Any], target_coverage: float) -> float:
    windows = [item for item in result.get("oot_windows", []) if isinstance(item, dict)]
    pnls = [float(item.get("net_profit") or 0.0) for item in windows]
    median_pnl = float(np.median(pnls)) if pnls else float(result.get("net_profit") or 0.0)
    worst_loss = max(0.0, -min(pnls)) if pnls else 0.0
    dd = float(result.get("max_drawdown_usdc") or 0.0)
    coverage_error = abs(float(result.get("coverage_pct") or 0.0) / 100.0 - target_coverage)
    return median_pnl - 0.5 * dd - 0.25 * worst_loss - 0.5 * coverage_error


class ThresholdEvaluatorFast:
    """Fast vectorized evaluator for joint thresholds."""

    def __init__(
        self,
        frame: pd.DataFrame,
        raw_scores: Iterable[float],
        calibrated_scores: Iterable[float],
        quotes: pd.DataFrame,
        *,
        strategy_branch: str = "COMBINED",
        min_edge: float = DEFAULT_MIN_EDGE,
        cost_buffer: float = DEFAULT_COST_BUFFER,
        fee_rate: float = DEFAULT_FEE_RATE,
        min_price: float = DEFAULT_MIN_PRICE,
        max_price: float = DEFAULT_MAX_PRICE,
        outsider_max_price: float = DEFAULT_OUTSIDER_MAX_PRICE,
        stake_usdc: float = DEFAULT_STAKE_USDC,
        slippage_pct: float = 0.0,
    ):
        self.branch = strategy_branch.strip().upper()
        self.stake_usdc = stake_usdc
        self.fee_rate = fee_rate

        raw = np.asarray(list(raw_scores), dtype=float)
        calibrated = np.asarray(list(calibrated_scores), dtype=float)
        if len(raw) != len(frame) or len(calibrated) != len(frame):
            raise ValueError("score arrays must align with frame")
        if stake_usdc <= 0:
            raise ValueError("stake_usdc must be positive")

        self.raw = raw
        self.valid_oof = np.isfinite(raw) & np.isfinite(calibrated)

        base = frame.reset_index(drop=True).copy()
        base["_orig_idx"] = np.arange(len(base))
        base["market_id"] = base["market_id"].astype(str)
        base["_p_up"] = calibrated
        self.n_markets = len(base)

        quote_frame = quotes.copy() if quotes is not None else pd.DataFrame()
        if quote_frame.empty:
            merged = base.iloc[0:0].copy()
        else:
            quote_frame["market_id"] = quote_frame["market_id"].astype(str)
            merged = base.merge(quote_frame.drop_duplicates("market_id", keep="first"), on="market_id", how="inner", suffixes=("", "_quote"))

        self.base_missing_quote = max(int(len(base) - len(merged)), 0)
        self.n_quotes = len(merged)

        def _entry_time_key(row):
            return str(row.get("recorded_at") or row.get("market_start") or "")

        if not merged.empty:
            merged["_sort_key"] = merged.apply(_entry_time_key, axis=1)
            merged.sort_values("_sort_key", kind="mergesort", inplace=True)
            merged.drop(columns=["_sort_key"], inplace=True)

        M = len(merged)
        self.orig_idx = merged["_orig_idx"].values
        self.valid_oof_m = self.valid_oof[self.orig_idx]

        self.up_eligible = np.zeros(M, dtype=bool)
        self.down_eligible = np.zeros(M, dtype=bool)
        self.up_pre_edge_eligible = np.zeros(M, dtype=bool)
        self.down_pre_edge_eligible = np.zeros(M, dtype=bool)

        self.up_reason = np.full(M, "", dtype=object)
        self.down_reason = np.full(M, "", dtype=object)

        self.trades_up = [None] * M
        self.trades_down = [None] * M

        self.up_pnl = np.zeros(M, dtype=float)
        self.down_pnl = np.zeros(M, dtype=float)
        self.up_won = np.zeros(M, dtype=bool)
        self.down_won = np.zeros(M, dtype=bool)

        for i, (_, row) in enumerate(merged.iterrows()):
            if not self.valid_oof_m[i]:
                continue

            p_up = _as_float(row.get("_p_up"))
            yes_ask, no_ask = _quote_prices(row)

            self._check_side(i, row, "UP", p_up, yes_ask, no_ask,
                             min_edge, cost_buffer, min_price, max_price,
                             outsider_max_price, slippage_pct)

            self._check_side(i, row, "DOWN", p_up, yes_ask, no_ask,
                             min_edge, cost_buffer, min_price, max_price,
                             outsider_max_price, slippage_pct)

    def _check_side(self, i: int, row: pd.Series, action: str, p_up: float | None, yes_ask: float | None, no_ask: float | None,
                    min_edge: float, cost_buffer: float, min_price: float, max_price: float,
                    outsider_max_price: float, slippage_pct: float):
        if p_up is None or not 0.0 <= p_up <= 1.0:
            self._set_reason(i, action, "invalid_calibrated_oof")
            return
        if yes_ask is None or no_ask is None:
            self._set_reason(i, action, "invalid_quote")
            return

        side = "BUY_YES" if action == "UP" else "BUY_NO"
        price = yes_ask if side == "BUY_YES" else no_ask
        outsider = (side == "BUY_YES" and yes_ask < 0.5) or (side == "BUY_NO" and yes_ask >= 0.5)

        if self.branch in {"OUTSIDER", "OUTSIDER_ONLY"} and not outsider:
            self._set_reason(i, action, "branch_filtered")
            return
        if self.branch in {"FAVORITE", "FAVORITE_ONLY"} and outsider:
            self._set_reason(i, action, "branch_filtered")
            return

        price_cap = outsider_max_price if self.branch in {"OUTSIDER", "OUTSIDER_ONLY"} else max_price
        if not min_price <= price <= min(max_price, price_cap):
            self._set_reason(i, action, "price_out_of_bounds")
            return

        p_win = p_up if side == "BUY_YES" else 1.0 - p_up
        gross_edge = float(p_win - price)
        net_edge = gross_edge - cost_buffer

        if action == "UP":
            self.up_pre_edge_eligible[i] = True
        else:
            self.down_pre_edge_eligible[i] = True

        if net_edge < min_edge:
            self._set_reason(i, action, "insufficient_edge")
            return

        executed_price = min(price * (1.0 + slippage_pct), 0.99)
        outcome = str(row.get("final_outcome") or ("YES" if row.get("target") == 1 else "NO"))
        won = (side == "BUY_YES" and outcome == "YES") or (side == "BUY_NO" and outcome == "NO")
        pnl = _pnl_for_trade(won=won, price=executed_price, stake_usdc=self.stake_usdc, fee_rate=self.fee_rate)

        trade = {
            "market_id": str(row["market_id"]),
            "asset": str(row.get("asset") or ""),
            "side": side,
            "action": action,
            "market_role": "OUTSIDER" if outsider else "FAVORITE",
            "price": float(executed_price),
            "p_win": float(p_win),
            "gross_edge": gross_edge,
            "net_edge": net_edge,
            "outcome": outcome,
            "won": bool(won),
            "pnl": pnl,
            "entry_time": row.get("recorded_at") or row.get("market_start"),
            "time_left_min": _as_float(row.get("time_left_min")),
            "price_bucket": _price_bucket(executed_price),
            "phase": _phase_bucket(_as_float(row.get("time_left_min")), row.get("vol_regime")),
        }

        if action == "UP":
            self.up_pnl[i] = pnl
            self.up_won[i] = won
            self.up_eligible[i] = True
            self.trades_up[i] = trade
        else:
            self.down_pnl[i] = pnl
            self.down_won[i] = won
            self.down_eligible[i] = True
            self.trades_down[i] = trade

    def _set_reason(self, i: int, action: str, reason: str):
        if action == "UP":
            self.up_reason[i] = reason
        else:
            self.down_reason[i] = reason

    def evaluate(self, lower: float, upper: float, summary_only: bool = False) -> dict[str, Any]:
        raw_m = self.raw[self.orig_idx]

        up_action_m = (raw_m >= upper) & self.valid_oof_m
        down_action_m = (raw_m <= lower) & self.valid_oof_m

        raw_valid = self.raw[self.valid_oof]
        n_valid = len(raw_valid)
        n_signals = int(((raw_valid >= upper) | (raw_valid <= lower)).sum())

        up_signals = int(up_action_m.sum())
        down_signals = int(down_action_m.sum())

        coverage_reasons = {"missing_oof": int((~self.valid_oof).sum()), "missing_quote": self.base_missing_quote}

        up_rejected = up_action_m & ~self.up_eligible
        down_rejected = down_action_m & ~self.down_eligible

        if up_rejected.any():
            reasons, counts = np.unique(self.up_reason[up_rejected], return_counts=True)
            for r, c in zip(reasons, counts):
                if r: coverage_reasons[r] = coverage_reasons.get(r, 0) + int(c)
        if down_rejected.any():
            reasons, counts = np.unique(self.down_reason[down_rejected], return_counts=True)
            for r, c in zip(reasons, counts):
                if r: coverage_reasons[r] = coverage_reasons.get(r, 0) + int(c)

        n_eligible = int((up_action_m & self.up_pre_edge_eligible).sum() + (down_action_m & self.down_pre_edge_eligible).sum())

        up_trades_mask = up_action_m & self.up_eligible
        down_trades_mask = down_action_m & self.down_eligible
        trade_mask = up_trades_mask | down_trades_mask

        n_trades = int(trade_mask.sum())
        invested = float(self.stake_usdc * n_trades)

        if n_trades > 0:
            valid_pnls = np.where(up_trades_mask, self.up_pnl, self.down_pnl)[trade_mask]
            valid_wons = np.where(up_trades_mask, self.up_won, self.down_won)[trade_mask]

            cumulative = np.cumsum(valid_pnls)
            peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
            max_dd = float(np.max(np.maximum(0.0, peak - cumulative)))
            max_dd_pct = max_dd / max(float(n_trades), 1.0) * 100.0

            win_rate = float(valid_wons.mean())
            net_profit = float(valid_pnls.sum())
        else:
            max_dd = 0.0
            max_dd_pct = 0.0
            win_rate = 0.0
            net_profit = 0.0

        res = {
            "lower_threshold": lower,
            "upper_threshold": upper,
            "strategy_branch": self.branch,
            "n_markets": self.n_markets,
            "n_quotes": self.n_quotes,
            "n_oof": n_valid,
            "n_signals": n_signals,
            "up_signals": up_signals,
            "down_signals": down_signals,
            "none_signals": max(n_valid - n_signals, 0),
            "coverage_pct": round(n_signals / n_valid * 100.0, 2) if n_valid else 0.0,
            "none_pct": round(max(n_valid - n_signals, 0) / n_valid * 100.0, 2) if n_valid else 0.0,
            "n_eligible": n_eligible,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "total_invested": invested,
            "stake_usdc": float(self.stake_usdc),
            "net_profit": net_profit,
            "roi_pct": float(net_profit / invested * 100.0) if invested else 0.0,
            "max_drawdown_usdc": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "coverage_reasons": coverage_reasons,
        }

        if summary_only:
            oot_windows = []
            if n_trades > 0:
                window_count = min(3, n_trades)
                for index, chunk in enumerate(np.array_split(valid_pnls, window_count), start=1):
                    if len(chunk) == 0:
                        continue
                    c_cum = np.cumsum(chunk)
                    c_peak = np.maximum.accumulate(np.maximum(c_cum, 0.0))
                    c_dd = float(np.max(np.maximum(0.0, c_peak - c_cum)))
                    chunk_invested = float(self.stake_usdc * len(chunk))
                    oot_windows.append({
                        "window": index,
                        "n_trades": int(len(chunk)),
                        "net_profit": float(chunk.sum()),
                        "max_drawdown_usdc": c_dd,
                        "max_drawdown_pct": c_dd / max(chunk_invested, 1e-9) * 100.0,
                    })
            res["oot_windows"] = oot_windows

        if not summary_only:
            trades = []
            if n_trades > 0:
                indices = np.nonzero(trade_mask)[0]
                for idx in indices:
                    if up_trades_mask[idx]:
                        trades.append(self.trades_up[idx])
                    else:
                        trades.append(self.trades_down[idx])

            res["slices"] = _slice_summary(trades, "action", self.stake_usdc) + _slice_summary(trades, "market_role", self.stake_usdc)
            res["oot_windows"] = _oot_window_summaries(trades, self.stake_usdc)
            res["trades"] = trades

        return res


def optimize_joint_thresholds(
    frame: pd.DataFrame,
    raw_scores: Iterable[float],
    calibrated_scores: Iterable[float],
    quotes: pd.DataFrame,
    *,
    target_coverages: Iterable[float] = TARGET_COVERAGES,
    selected_target_coverage: float = 0.40,
    strategy_branches: Iterable[str] = ("OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"),
    **backtest_options: Any,
) -> dict[str, Any]:
    raw = np.asarray(list(raw_scores), dtype=float)
    calibrated = np.asarray(list(calibrated_scores), dtype=float)
    if len(raw) != len(frame) or len(calibrated) != len(frame):
        raise ValueError("score arrays must align with frame")
    target = float(selected_target_coverage)
    if not 0.0 < target < 1.0:
        raise ValueError("selected target coverage must be between 0 and 1")

    supported_kwargs = {"min_edge", "cost_buffer", "fee_rate", "min_price", "max_price", "outsider_max_price", "stake_usdc", "slippage_pct"}
    can_fast_path = all(k in supported_kwargs for k in backtest_options)

    sweep: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for coverage in target_coverages:
        coverage = float(coverage)
        for branch in strategy_branches:
            evaluator = None
            if can_fast_path:
                evaluator = ThresholdEvaluatorFast(
                    frame, raw, calibrated, quotes, strategy_branch=branch, **backtest_options
                )
            branch_candidates: list[dict[str, Any]] = []
            for pair in threshold_grid(raw, target_coverage=coverage):
                if can_fast_path and evaluator is not None:
                    result = evaluator.evaluate(pair.lower, pair.upper, summary_only=True)
                else:
                    result = evaluate_thresholded_polymarket(
                        frame, raw, calibrated, quotes,
                        lower=pair.lower, upper=pair.upper, strategy_branch=branch,
                        summary_only=True,
                        **backtest_options,
                    )
                summary = {key: value for key, value in result.items() if key not in ("trades", "slices", "oot_windows")}
                summary["target_coverage_pct"] = round(coverage * 100.0, 2)
                summary["economic_score"] = round(_economic_score(result, coverage), 8)
                branch_candidates.append(summary)
                candidates.append({"target": coverage, "branch": branch, "pair": pair, "result": result})
            if branch_candidates:
                sweep.append(max(branch_candidates, key=lambda item: (item["economic_score"], item.get("net_profit", 0.0), item.get("n_trades", 0))))

    selected = [item for item in candidates if abs(item["target"] - target) < 1e-9] or candidates
    if not selected:
        pair = ThresholdPair(0.49, 0.51)
        branch = "COMBINED"
    else:
        selected_item = max(selected, key=lambda item: (_economic_score(item["result"], target), item["result"].get("net_profit", 0.0), item["result"].get("n_trades", 0)))
        pair = selected_item["pair"]
        branch = selected_item["branch"]

    if can_fast_path:
        eval_comb = ThresholdEvaluatorFast(frame, raw, calibrated, quotes, strategy_branch=branch, **backtest_options)
        final_result = eval_comb.evaluate(pair.lower, pair.upper, summary_only=False)
    else:
        final_result = evaluate_thresholded_polymarket(frame, raw, calibrated, quotes, lower=pair.lower, upper=pair.upper, strategy_branch=branch, summary_only=False, **backtest_options)

    selected_result = {key: value for key, value in final_result.items() if key != "trades"}
    selected_result["target_coverage_pct"] = round(target * 100.0, 2)
    selected_result["economic_score"] = round(_economic_score(final_result, target), 8)
    return {
        "selected": selected_result,
        "selected_lower_threshold": pair.lower,
        "selected_upper_threshold": pair.upper,
        "selected_target_coverage": target,
        "sweep": sweep,
    }
