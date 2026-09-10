"""Step 18: two sample levels.

- Forecast level: ONE row per market with P(UP) target (actual_outcome UP=1).
  Used for the main model comparison table; all compared models share the
  SAME rows (common coverage); coverage losses shown separately.
- Trading level: observed UP/DOWN quotes + execution data per opportunity.

UP and DOWN legs of one contract must never become two independent
training examples. Model comparison must not win on different markets.
"""
from __future__ import annotations

from dataclasses import dataclass


FORECAST_COLUMNS = (
    "market_id", "asset", "decision_at", "end_at",
    "p_up_market", "down_mid_diag",
    "z", "time_left_sec", "sigma_per_sec",
    "ret_1m", "ret_3m", "ret_5m", "n_cross_5m", "secs_since_cross",
    "d_norm_1m", "range_pos_5m", "bounce_from_min", "pullback_from_max",
    "ct_signal", "ct_regime",
    "up_bid", "up_ask", "down_bid", "down_ask",
    "target_up", "strike_source",
)

M1_COLUMNS = ("z", "time_left_sec", "sigma_per_sec")
M2_COLUMNS = M1_COLUMNS + (
    "ret_1m", "ret_3m", "ret_5m", "n_cross_5m", "secs_since_cross",
    "d_norm_1m", "range_pos_5m", "bounce_from_min", "pullback_from_max",
)


@dataclass(frozen=True)
class CoverageReport:
    n_forecast_rows: int
    n_common_rows: int
    dropped: dict  # reason -> count


def common_rows(rows: list[dict], required: tuple[str, ...] = M2_COLUMNS) -> tuple[list[dict], CoverageReport]:
    """Keep rows where ALL compared models have features; report the rest."""
    from collections import Counter
    dropped: Counter[str] = Counter()
    keep = []
    seen = set()
    for r in rows:
        mid = r.get("market_id")
        if mid in seen:
            dropped["duplicate_market_id"] += 1
            continue
        seen.add(mid)
        # UP/DOWN legs must already be collapsed: reject leg-level rows
        if r.get("leg") in ("UP", "DOWN") and r.get("target_up") is None:
            dropped["leg_level_row"] += 1
            continue
        missing = [c for c in required if r.get(c) is None]
        # secs_since_cross may legitimately be None (no crossing) -> encode later
        missing = [c for c in missing if c != "secs_since_cross"]
        if missing:
            dropped[f"missing:{','.join(sorted(missing))}"] += 1
            continue
        keep.append(r)
    return keep, CoverageReport(len(rows), len(keep), dict(dropped))
