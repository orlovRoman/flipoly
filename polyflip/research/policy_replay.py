"""Causal, all-opportunity replay for dashboard policy audits.

This module deliberately does not train models or search thresholds.  The
input must contain one row per opportunity (including rows that were skipped),
and every fixed variant is evaluated on exactly that same row set.  Missing
quotes or predictions remain ``INSUFFICIENT_DATA`` and are never converted to
zero PnL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from polyflip.trading.fee_model import fee_per_share


REQUIRED_COLUMNS = {
    "opportunity_id",
    "market_id",
    "asset",
    "decision_at",
    "time_left_sec",
    "yes_ask",
    "no_ask",
    "yes_mid",
    "no_mid",
    "p_model_yes",
    "final_outcome",
}


@dataclass(frozen=True)
class ReplayConfig:
    name: str
    allow_favorite: bool = True
    allow_outsider: bool = True
    min_time_left_sec: float = 210.0
    max_time_left_sec: float = 900.0
    min_edge: float = 0.03
    budget_usdc: float = 1.0
    slippage_rate: float = 0.005
    fee_rate: float = 0.07
    fee_exponent: float = 1.0


FIXED_VARIANTS: tuple[ReplayConfig, ...] = (
    ReplayConfig("CURRENT"),
    ReplayConfig("STRICT", min_time_left_sec=210.0, max_time_left_sec=300.0, min_edge=0.05),
    ReplayConfig("FAVORITE_ONLY", allow_outsider=False),
    ReplayConfig("OUTSIDER_ONLY", allow_favorite=False),
)


def _float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _outcome(value: Any) -> int | None:
    text = str(value or "").strip().upper()
    if text in {"YES", "UP", "1", "TRUE", "WIN"}:
        return 1
    if text in {"NO", "DOWN", "0", "FALSE", "LOSS"}:
        return 0
    return None


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("opportunity ledger is empty")
    for index, row in enumerate(rows):
        missing = REQUIRED_COLUMNS - set(row.keys())
        if missing:
            raise ValueError(
                f"opportunity ledger row {index} missing columns: {sorted(missing)}"
            )
    ids = [str(row.get("opportunity_id")) for row in rows]
    if any(not item or item == "None" for item in ids):
        raise ValueError("every row must have an opportunity_id")
    if len(ids) != len(set(ids)):
        raise ValueError("opportunity_id must be unique")
    for row in rows:
        decision_at = row.get("decision_at")
        label_at = row.get("label_available_at")
        try:
            decision_dt = datetime.fromisoformat(str(decision_at).replace("Z", "+00:00"))
            if decision_dt.tzinfo is None:
                raise ValueError("decision_at must include a timezone")
            if label_at not in (None, ""):
                label_dt = datetime.fromisoformat(str(label_at).replace("Z", "+00:00"))
                if label_dt.tzinfo is None:
                    raise ValueError("label_available_at must include a timezone")
                if label_dt < decision_dt:
                    raise ValueError("label_available_at precedes decision_at")
        except (TypeError, ValueError) as exc:
            if "precedes" in str(exc):
                raise
            raise ValueError(f"invalid causal timestamp in row {row.get('opportunity_id')}") from exc


def _evaluate_one(row: Mapping[str, Any], cfg: ReplayConfig) -> dict[str, Any]:
    base = {
        "opportunity_id": str(row["opportunity_id"]),
        "market_id": str(row["market_id"]),
        "asset": str(row.get("asset") or "").upper(),
        "variant": cfg.name,
        "status": "SKIP",
        "reason": None,
        "role": None,
        "side": None,
        "ask": None,
        "p_win": None,
        "edge": None,
        "shares": None,
        "gross_pnl": None,
        "fee_usdc": None,
        "slippage_usdc": None,
        "net_pnl": None,
    }
    required_values = [
        _float(row, key)
        for key in ("time_left_sec", "yes_ask", "no_ask", "yes_mid", "no_mid", "p_model_yes")
    ]
    if any(value is None for value in required_values):
        base["status"], base["reason"] = "INSUFFICIENT_DATA", "MISSING_CAUSAL_INPUT"
        return base
    time_left, yes_ask, no_ask, yes_mid, no_mid, p_yes = required_values
    assert time_left is not None and yes_ask is not None and no_ask is not None
    assert yes_mid is not None and no_mid is not None and p_yes is not None
    if not (cfg.min_time_left_sec <= time_left <= cfg.max_time_left_sec):
        base["reason"] = "OUTSIDE_TIME_WINDOW"
        return base
    if not all(0.0 < value < 1.0 for value in (yes_ask, no_ask, yes_mid, no_mid)):
        base["status"], base["reason"] = "INSUFFICIENT_DATA", "INVALID_CAUSAL_QUOTE"
        return base
    side = "YES" if p_yes >= 0.5 else "NO"
    ask = yes_ask if side == "YES" else no_ask
    mid = yes_mid if side == "YES" else no_mid
    p_win = p_yes if side == "YES" else 1.0 - p_yes
    role = "FAVORITE" if mid >= 0.5 else "OUTSIDER"
    base.update({"role": role, "side": side, "ask": ask, "p_win": p_win})
    if role == "FAVORITE" and not cfg.allow_favorite:
        base["reason"] = "ROLE_DISABLED"
        return base
    if role == "OUTSIDER" and not cfg.allow_outsider:
        base["reason"] = "ROLE_DISABLED"
        return base
    fee = float(fee_per_share(ask, fee_rate=cfg.fee_rate, fee_exponent=cfg.fee_exponent))
    execution_price = ask * (1.0 + cfg.slippage_rate)
    edge = p_win - execution_price - fee
    base["edge"] = edge
    if edge < cfg.min_edge:
        base["reason"] = "EDGE_BELOW_MIN"
        return base
    outcome = _outcome(row.get("final_outcome"))
    if outcome is None:
        base["status"], base["reason"] = "INSUFFICIENT_DATA", "UNKNOWN_OUTCOME"
        return base
    shares = cfg.budget_usdc / execution_price
    fee_usdc = shares * float(fee_per_share(execution_price, fee_rate=cfg.fee_rate, fee_exponent=cfg.fee_exponent))
    gross = shares * (outcome - execution_price)
    base.update(
        {
            "status": "BUY",
            "reason": "ACCEPTED",
            "shares": shares,
            "gross_pnl": gross,
            "fee_usdc": fee_usdc,
            "slippage_usdc": shares * (execution_price - ask),
            "net_pnl": gross - fee_usdc,
        }
    )
    return base


def replay_opportunities(
    rows: Iterable[Mapping[str, Any]],
    variants: Sequence[ReplayConfig] = FIXED_VARIANTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay all rows under fixed variants and return rows plus aggregates."""
    row_list = [dict(row) for row in rows]
    _validate_rows(row_list)
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"opportunities": len(row_list), "variants": {}}
    for cfg in variants:
        variant_rows = [_evaluate_one(row, cfg) for row in row_list]
        results.extend(variant_rows)
        buys = [row for row in variant_rows if row["status"] == "BUY"]
        counts: dict[str, int] = {}
        for row in variant_rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        summary["variants"][cfg.name] = {
            "config": asdict(cfg),
            "status_counts": counts,
            "buy_count": len(buys),
            "net_pnl": sum(float(row["net_pnl"] or 0.0) for row in buys),
            "gross_pnl": sum(float(row["gross_pnl"] or 0.0) for row in buys),
            "fee_usdc": sum(float(row["fee_usdc"] or 0.0) for row in buys),
            "missing_not_zero": sum(row["status"] == "INSUFFICIENT_DATA" for row in variant_rows),
        }
        if sum(counts.values()) != len(row_list):
            raise AssertionError(f"status partition failed for {cfg.name}")
    summary["same_opportunity_set"] = all(
        len([row for row in results if row["variant"] == cfg.name]) == len(row_list)
        for cfg in variants
    )
    return results, summary
